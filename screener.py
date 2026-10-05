"""
KRM Stocks Screener
يشتغل كل يوم تلقائياً عبر GitHub Actions
يسحب بيانات من yfinance، يرشح الأسهم، ويحفظ في Firebase Firestore
"""

import json
import os
from datetime import datetime, timezone
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore

# ─── Firebase Init ───────────────────────────────────────────────
FIREBASE_CREDS = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(FIREBASE_CREDS)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ─── أسهم المراقبة ───────────────────────────────────────────────
# الأسهم الأمريكية (غير رسمية - قائمة يدوية حتى يُضاف فلتر TradingView)
US_WATCHLIST = [
    "EVLV", "PHAT", "SOUN", "ABAT", "LZ",
    "LFMD", "BW", "MARA", "RIOT", "CLSK",
    "CIFR", "WULF", "HIVE", "BTBT", "BSRT",
    "ACHR", "JOBY", "LILM", "EOSE", "BLNK",
    "CHPT", "NKLA", "HYLN", "FSR", "ARVL",
    "SPCE", "RKLB", "ASTR", "MNTS", "ASTS",
    "LUNR", "RDW", "VORB", "SFET", "FLNC",
    "BE", "FCEL", "PLUG", "BLDP", "NFGC",
]

# السعودي - رموز تداول (بدون .SR)
SA_WATCHLIST = [
    "1832", "1140", "4261", "3008", "4321",
    "2010", "2310", "2084", "1213", "1201",
    "4280", "4240", "4200", "2382", "4164",
    "1810", "2222", "2330", "4030", "4160",
]

# ─── إعدادات الفلتر ──────────────────────────────────────────────
PERIOD_DAYS    = 220    # نافذة الحساب
MAX_SCORE      = 0.10   # أقصى نسبة قرب من القاع (10%)
HOURLY_DROP    = -0.10  # أقصى تراجع ساعي مسموح (-10%)
MIN_VOLUME     = 100_000  # حجم تداول أدنى (US)
TOP_N          = 10     # أفضل N سهم لكل سوق

# ─── دالة حساب السكور ────────────────────────────────────────────
def proximity_score(hist):
    """(close - period_low) / (period_high - period_low)"""
    if hist.empty or len(hist) < 20:
        return None
    period_high = hist["High"].max()
    period_low  = hist["Low"].min()
    if period_high == period_low:
        return None
    close = hist["Close"].iloc[-1]
    return round((close - period_low) / (period_high - period_low), 4)

def hourly_drop(ticker_obj):
    """أقل إغلاق خلال 72 ساعة → نسبة التراجع من الأعلى"""
    try:
        h = ticker_obj.history(period="4d", interval="1h")
        if h.empty:
            return 0.0
        high_72 = h["High"].max()
        last    = h["Close"].iloc[-1]
        return round((last - high_72) / high_72, 4)
    except:
        return 0.0

def build_candles(hist):
    """تحويل DataFrame إلى قائمة OHLCV للشارت"""
    candles = []
    for idx, row in hist.iterrows():
        try:
            t = int(idx.timestamp())
            candles.append({
                "t": t,
                "o": round(float(row["Open"]), 4),
                "h": round(float(row["High"]), 4),
                "l": round(float(row["Low"]), 4),
                "c": round(float(row["Close"]), 4),
                "v": int(row["Volume"]),
            })
        except:
            pass
    return candles[-120:]  # آخر 120 شمعة يومية فقط

# ─── فحص سهم واحد ────────────────────────────────────────────────
def screen_stock(symbol, market="us"):
    tv_symbol = f"{symbol}.SR" if market == "sa" else symbol
    try:
        t = yf.Ticker(tv_symbol)
        hist = t.history(period=f"{PERIOD_DAYS}d", interval="1d")
        if hist.empty or len(hist) < 20:
            print(f"  ⚠️  {symbol}: لا توجد بيانات كافية")
            return None

        # فحص الحجم (US فقط)
        if market == "us":
            avg_vol = hist["Volume"].tail(20).mean()
            if avg_vol < MIN_VOLUME:
                print(f"  ❌ {symbol}: حجم منخفض ({int(avg_vol):,})")
                return None

        score = proximity_score(hist)
        if score is None or score > MAX_SCORE:
            print(f"  ❌ {symbol}: سكور {score}")
            return None

        drop = hourly_drop(t)
        if drop < HOURLY_DROP:
            print(f"  ❌ {symbol}: تراجع ساعي {drop*100:.1f}%")
            return None

        note = build_note(score, drop, market)
        candles = build_candles(hist)
        close = round(float(hist["Close"].iloc[-1]), 4)

        print(f"  ✅ {symbol}: سكور={score:.3f} تراجع={drop*100:.1f}%")
        return {
            "score": score,
            "drop72h": drop,
            "close": close,
            "note": note,
            "candles": candles,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    except Exception as e:
        print(f"  ⚠️  {symbol}: خطأ {e}")
        return None

def build_note(score, drop, market):
    parts = []
    if score < 0.03:
        parts.append("قريب جداً من القاع")
    elif score < 0.06:
        parts.append("قريب من القاع")
    else:
        parts.append("تصحيح معقول")

    if drop < -0.07:
        parts.append(f"تراجع ساعي {drop*100:.1f}%")
    elif drop < -0.03:
        parts.append("تراجع ساعي خفيف")
    else:
        parts.append("مستقر ساعياً")

    return "، ".join(parts)

# ─── فرز وحفظ ────────────────────────────────────────────────────
def run_market(watchlist, collection_name, market_label, market_code):
    print(f"\n{'='*40}")
    print(f"🔍 فرز {market_label} ({len(watchlist)} سهم)")
    print(f"{'='*40}")

    results = {}
    for sym in watchlist:
        print(f"\n▶ {sym}")
        data = screen_stock(sym, market_code)
        if data:
            results[sym] = data

    # أفضل N
    top = sorted(results.items(), key=lambda x: x[1]["score"])[:TOP_N]
    print(f"\n✅ مؤهل: {len(top)} سهم من {len(watchlist)}")

    # اقرأ الإصدار الحالي
    meta_ref = db.collection("meta").document(collection_name)
    meta_doc = meta_ref.get()
    version = (meta_doc.to_dict() or {}).get("version", 0) + 1

    # احذف القديم واحفظ الجديد
    col_ref = db.collection(collection_name)
    old_docs = col_ref.stream()
    for d in old_docs:
        d.reference.delete()

    batch = db.batch()
    for sym, data in top:
        batch.set(col_ref.document(sym), data)
    batch.set(meta_ref, {
        "version": version,
        "count": len(top),
        "lastRun": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "market": market_label,
    })
    batch.commit()
    print(f"✅ حُفظ في Firestore ({collection_name}) - الإصدار {version}")
    return top

# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 بدء الفرز اليومي")
    print(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    us_top  = run_market(US_WATCHLIST, "suggestions_us", "أمريكي",  "us")
    sa_top  = run_market(SA_WATCHLIST, "suggestions_sa", "سعودي", "sa")

    print("\n" + "="*40)
    print("📊 ملخص النتائج")
    print("="*40)
    print(f"🇺🇸 أمريكي: {[s for s,_ in us_top]}")
    print(f"🇸🇦 سعودي:  {[s for s,_ in sa_top]}")
    print("\n✅ انتهى الفرز")
