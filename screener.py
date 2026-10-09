"""
KRM Stocks Screener - v2
ثلاث طبقات: كلاسيكي + Wyckoff + Elliott
+ فرز الارتكاز (نطاق ضيق عند القاع)
يشتغل كل يوم تلقائياً عبر GitHub Actions
"""

import json
import os
from datetime import datetime, timezone
import numpy as np
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore

# ─── Firebase Init ───────────────────────────────────────────────
FIREBASE_CREDS = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(FIREBASE_CREDS)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ─── أسهم المراقبة ───────────────────────────────────────────────
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

SA_WATCHLIST = [
    "1832", "1140", "4261", "3008", "4321",
    "2010", "2310", "2084", "1213", "1201",
    "4280", "4240", "4200", "2382", "4164",
    "1810", "2222", "2330", "4030", "4160",
]

# ─── إعدادات ─────────────────────────────────────────────────────
PERIOD_DAYS   = 220
MAX_SCORE     = 0.10
HOURLY_DROP   = -0.10
MIN_VOLUME    = 100_000
TOP_N         = 999

# ─── طبقة 1: كلاسيكي ─────────────────────────────────────────────
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
    try:
        h = ticker_obj.history(period="4d", interval="1h")
        if h.empty:
            return 0.0
        high_72 = h["High"].max()
        last    = h["Close"].iloc[-1]
        return round((last - high_72) / high_72, 4)
    except:
        return 0.0

# ─── طبقة 2: Wyckoff ─────────────────────────────────────────────
def wyckoff_score(hist):
    """
    يكشف التراكم الصامت:
    - حجم متناقص عند القاع (Accumulation)
    - سعر يتضيق (Spring احتمالي)
    يرجع: 0-1 (كلما كان أعلى = تراكم أقوى)
    """
    if len(hist) < 40:
        return 0.0

    closes  = hist["Close"].values
    volumes = hist["Volume"].values

    # آخر 20 يوم مقارنة بـ 20 يوم قبلها
    recent_vol = volumes[-20:].mean()
    prev_vol   = volumes[-40:-20].mean()

    # انخفاض الحجم = تراكم صامت (مو تصريف بحجم عالي)
    vol_declining = recent_vol < prev_vol * 0.85

    # نطاق السعر ضيق آخر 10 أيام
    recent_range = (closes[-10:].max() - closes[-10:].min()) / closes[-10:].mean()
    tight_range  = recent_range < 0.06  # أقل من 6% تذبذب

    # السعر فوق أدنى نقطة (مو يكسر القاع)
    period_low   = hist["Low"].min()
    above_low    = closes[-1] > period_low * 1.02

    score = 0.0
    if vol_declining: score += 0.4
    if tight_range:   score += 0.4
    if above_low:     score += 0.2

    return round(score, 2)

# ─── طبقة 3: Elliott Wave (مبسّط) ───────────────────────────────
def elliott_score(hist):
    """
    يكشف موجة تصحيح (2 أو 4):
    - هبوط من قمة واضحة
    - التصحيح لم يكسر قاع الموجة 1
    - Fibonacci retracement 38-78%
    يرجع: 0-1
    """
    if len(hist) < 60:
        return 0.0

    closes = hist["Close"].values

    # ابحث عن آخر قمة واضحة (أعلى نقطة في آخر 60 يوم)
    peak_idx = np.argmax(closes[-60:])
    peak_val = closes[-60:][peak_idx]
    current  = closes[-1]

    # لازم يكون في تصحيح (السعر أقل من القمة)
    if current >= peak_val * 0.95:
        return 0.0

    # حساب نسبة التصحيح من القمة
    # ابحث عن قاع قبل القمة
    if peak_idx < 5:
        return 0.0

    base_val = closes[-60:][:peak_idx].min()
    if peak_val <= base_val:
        return 0.0

    wave_size    = peak_val - base_val
    correction   = peak_val - current
    retrace_pct  = correction / wave_size

    score = 0.0

    # Fibonacci 38.2% - 78.6% = منطقة التصحيح الصحية
    if 0.382 <= retrace_pct <= 0.786:
        score += 0.6
        # 50-61.8% = المنطقة المثالية
        if 0.50 <= retrace_pct <= 0.618:
            score += 0.4

    return round(min(score, 1.0), 2)

# ─── فرز الارتكاز ────────────────────────────────────────────────
def is_consolidating(hist):
    """
    معيار فيصل للارتكاز:
    - نطاق ضيق آخر 7 أيام (أقل من 8% تذبذب)
    - حجم منخفض (نائم)
    - عند الدعم (قريب من القاع)
    - أداء شهري سلبي أو محايد
    """
    if len(hist) < 30:
        return False, ""

    closes  = hist["Close"].values
    volumes = hist["Volume"].values

    # نطاق ضيق آخر 7 أيام
    recent_high  = hist["High"].values[-7:].max()
    recent_low   = hist["Low"].values[-7:].min()
    range_pct    = (recent_high - recent_low) / closes[-7:].mean()
    tight        = range_pct < 0.08

    # حجم منخفض (آخر 7 أيام أقل من المتوسط)
    avg_vol_20   = volumes[-20:].mean()
    recent_vol7  = volumes[-7:].mean()
    quiet        = recent_vol7 < avg_vol_20 * 0.85

    # عند الدعم (آخر سعر ضمن أدنى 20% من النطاق الشهري)
    month_high   = hist["High"].values[-22:].max()
    month_low    = hist["Low"].values[-22:].min()
    month_range  = month_high - month_low
    if month_range == 0:
        return False, ""
    pos_in_range = (closes[-1] - month_low) / month_range
    near_support = pos_in_range < 0.25

    # أداء شهري سلبي أو محايد
    month_perf   = (closes[-1] - closes[-22]) / closes[-22] if len(closes) >= 22 else 0
    monthly_down = month_perf <= 0.03

    reasons = []
    if tight:        reasons.append("نطاق ضيق")
    if quiet:        reasons.append("حجم هادئ")
    if near_support: reasons.append("عند الدعم")
    if monthly_down: reasons.append("أداء محايد")

    is_cons = tight and quiet and near_support
    return is_cons, "، ".join(reasons)

# ─── بناء الشموع ─────────────────────────────────────────────────
def build_candles(hist):
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
    return candles[-120:]

# ─── بناء الملاحظة ────────────────────────────────────────────────
def build_note(score, drop, wyckoff, elliott, consolidating, cons_reason):
    parts = []

    # كلاسيكي
    if score < 0.03:
        parts.append("قريب جداً من القاع")
    elif score < 0.06:
        parts.append("قريب من القاع")
    else:
        parts.append("تصحيح معقول")

    # Wyckoff
    if wyckoff >= 0.8:
        parts.append("تراكم قوي 🟢")
    elif wyckoff >= 0.6:
        parts.append("إشارات تراكم")

    # Elliott
    if elliott >= 0.8:
        parts.append("تصحيح فيبوناتشي مثالي 📐")
    elif elliott >= 0.6:
        parts.append("تصحيح صحي")

    # ساعي
    if drop < -0.07:
        parts.append(f"تراجع {drop*100:.1f}%")
    elif abs(drop) < 0.02:
        parts.append("مستقر")

    return "، ".join(parts)

# ─── حساب درجة الثقة الكلية ──────────────────────────────────────
def confidence_score(prox, wyckoff, elliott):
    """
    0-100: كلما ارتفع = فرصة أقوى
    """
    # عكس proximity (كلما انخفض = أفضل)
    prox_score = max(0, 1 - (prox / 0.10))  # 0→1, 0.10→0

    total = (prox_score * 0.40) + (wyckoff * 0.35) + (elliott * 0.25)
    return round(total * 100)

# ─── فحص سهم واحد ────────────────────────────────────────────────
def screen_stock(symbol, market="us"):
    tv_symbol = f"{symbol}.SR" if market == "sa" else symbol
    try:
        t    = yf.Ticker(tv_symbol)
        hist = t.history(period=f"{PERIOD_DAYS}d", interval="1d")

        if hist.empty or len(hist) < 20:
            print(f"  ⚠️  {symbol}: لا بيانات")
            return None

        # حجم (US فقط)
        if market == "us":
            avg_vol = hist["Volume"].tail(20).mean()
            if avg_vol < MIN_VOLUME:
                print(f"  ❌ {symbol}: حجم منخفض")
                return None

        # ── الطبقة 1: كلاسيكي ──
        score = proximity_score(hist)
        if score is None or score > MAX_SCORE:
            print(f"  ❌ {symbol}: سكور {score}")
            return None

        drop = hourly_drop(t)
        if drop < HOURLY_DROP:
            print(f"  ❌ {symbol}: تراجع ساعي {drop*100:.1f}%")
            return None

        # ── الطبقة 2: Wyckoff ──
        wyckoff = wyckoff_score(hist)

        # ── الطبقة 3: Elliott ──
        elliott = elliott_score(hist)

        # ── الارتكاز ──
consolidating, cons_reason = is_consolidating(hist)
consolidating = bool(consolidating)
        # ── درجة الثقة ──
        confidence = confidence_score(score, wyckoff, elliott)

        note    = build_note(score, drop, wyckoff, elliott, consolidating, cons_reason)
        candles = build_candles(hist)
        close   = round(float(hist["Close"].iloc[-1]), 4)

        print(f"  ✅ {symbol}: سكور={score:.3f} | ثقة={confidence}% | Wyckoff={wyckoff} | Elliott={elliott} | ارتكاز={'✓' if consolidating else '✗'}")

        return {
            "score":         score,
            "confidence":    confidence,
            "wyckoff":       wyckoff,
            "elliott":       elliott,
            "consolidating": consolidating,
            "consReason":    cons_reason,
            "drop72h":       drop,
            "close":         close,
            "note":          note,
            "candles":       candles,
            "date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

    except Exception as e:
        print(f"  ⚠️  {symbol}: خطأ {e}")
        return None

# ─── فرز وحفظ ────────────────────────────────────────────────────
def run_market(watchlist, collection_us, collection_cons, market_label, market_code):
    print(f"\n{'='*40}")
    print(f"🔍 فرز {market_label} ({len(watchlist)} سهم)")
    print(f"{'='*40}")

    results       = {}
    consolidating = {}

    for sym in watchlist:
        print(f"\n▶ {sym}")
        data = screen_stock(sym, market_code)
        if data:
            results[sym] = data
            if data["consolidating"]:
                consolidating[sym] = data

    # أفضل N حسب الثقة
    top      = sorted(results.items(), key=lambda x: -x[1]["confidence"])[:TOP_N]
    top_cons = sorted(consolidating.items(), key=lambda x: -x[1]["confidence"])[:TOP_N]

    print(f"\n✅ فرص: {len(top)} | ارتكاز: {len(top_cons)}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── حفظ الفرص ──
    meta_ref = db.collection("meta").document(collection_us)
    meta_doc = meta_ref.get()
    version  = (meta_doc.to_dict() or {}).get("version", 0) + 1

    col_ref  = db.collection(collection_us)
    for d in col_ref.stream():
        d.reference.delete()

    batch = db.batch()
    for sym, data in top:
        batch.set(col_ref.document(sym), data)
    batch.set(meta_ref, {
        "version": version,
        "count":   len(top),
        "lastRun": today,
        "market":  market_label,
    })
    batch.commit()
    print(f"✅ فرص → Firestore ({collection_us}) v{version}")

    # ── حفظ الارتكاز ──
    meta_cons = db.collection("meta").document(collection_cons)
    meta_cons_doc = meta_cons.get()
    version_cons  = (meta_cons_doc.to_dict() or {}).get("version", 0) + 1

    col_cons = db.collection(collection_cons)
    for d in col_cons.stream():
        d.reference.delete()

    batch2 = db.batch()
    for sym, data in top_cons:
        batch2.set(col_cons.document(sym), data)
    batch2.set(meta_cons, {
        "version": version_cons,
        "count":   len(top_cons),
        "lastRun": today,
        "market":  market_label,
    })
    batch2.commit()
    print(f"✅ ارتكاز → Firestore ({collection_cons}) v{version_cons}")

    return top, top_cons

# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 بدء الفرز اليومي - v2")
    print(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    us_top, us_cons = run_market(
        US_WATCHLIST,
        "suggestions_us", "consolidation_us",
        "أمريكي", "us"
    )
    sa_top, sa_cons = run_market(
        SA_WATCHLIST,
        "suggestions_sa", "consolidation_sa",
        "سعودي", "sa"
    )

    print("\n" + "="*40)
    print("📊 ملخص")
    print("="*40)
    print(f"🇺🇸 فرص: {[s for s,_ in us_top]}")
    print(f"🇺🇸 ارتكاز: {[s for s,_ in us_cons]}")
    print(f"🇸🇦 فرص: {[s for s,_ in sa_top]}")
    print(f"🇸🇦 ارتكاز: {[s for s,_ in sa_cons]}")
    print("\n✅ انتهى الفرز")
