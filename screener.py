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
    if len(hist) < 40:
        return 0.0
    closes  = hist["Close"].values
    volumes = hist["Volume"].values
    recent_vol = volumes[-20:].mean()
    prev_vol   = volumes[-40:-20].mean()
    vol_declining = recent_vol < prev_vol * 0.85
    recent_range = (closes[-10:].max() - closes[-10:].min()) / closes[-10:].mean()
    tight_range  = recent_range < 0.06
    period_low   = hist["Low"].min()
    above_low    = closes[-1] > period_low * 1.02
    score = 0.0
    if vol_declining: score += 0.4
    if tight_range:   score += 0.4
    if above_low:     score += 0.2
    return round(score, 2)

# ─── طبقة 3: Elliott Wave ────────────────────────────────────────
def elliott_score(hist):
    if len(hist) < 60:
        return 0.0
    closes = hist["Close"].values
    peak_idx = np.argmax(closes[-60:])
    peak_val = closes[-60:][peak_idx]
    current  = closes[-1]
    if current >= peak_val * 0.95:
        return 0.0
    if peak_idx < 5:
        return 0.0
    base_val = closes[-60:][:peak_idx].min()
    if peak_val <= base_val:
        return 0.0
    wave_size   = peak_val - base_val
    correction  = peak_val - current
    retrace_pct = correction / wave_size
    score = 0.0
    if 0.382 <= retrace_pct <= 0.786:
        score += 0.6
        if 0.50 <= retrace_pct <= 0.618:
            score += 0.4
    return round(min(score, 1.0), 2)

# ─── فرز الارتكاز ────────────────────────────────────────────────
def is_consolidating(hist):
    if len(hist) < 30:
        return False, ""
    closes  = hist["Close"].values
    volumes = hist["Volume"].values
    recent_high  = hist["High"].values[-7:].max()
    recent_low   = hist["Low"].values[-7:].min()
    range_pct    = (recent_high - recent_low) / closes[-7:].mean()
    tight        = bool(range_pct < 0.08)
    avg_vol_20   = volumes[-20:].mean()
    recent_vol7  = volumes[-7:].mean()
    quiet        = bool(recent_vol7 < avg_vol_20 * 0.85)
    month_high   = hist["High"].values[-22:].max()
    month_low    = hist["Low"].values[-22:].min()
    month_range  = month_high - month_low
    if month_range == 0:
        return False, ""
    pos_in_range = (closes[-1] - month_low) / month_range
    near_support = bool(pos_in_range < 0.25)
    month_perf   = (closes[-1] - closes[-22]) / closes[-22] if len(closes) >= 22 else 0
    monthly_down = bool(month_perf <= 0.03)
    reasons = []
    if tight:        reasons.append("نطاق ضيق")
    if quiet:        reasons.append("حجم هادئ")
    if near_support: reasons.append("عند الدعم")
    if monthly_down: reasons.append("أداء محايد")
    is_cons = tight and quiet and near_support
    return bool(is_cons), "، ".join(reasons)

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
    if score < 0.03:
        parts.append("قريب جداً من القاع")
    elif score < 0.06:
        parts.append("قريب من القاع")
    else:
        parts.append("تصحيح معقول")
    if wyckoff >= 0.8:
        parts.append("تراكم قوي 🟢")
    elif wyckoff >= 0.6:
        parts.append("إشارات تراكم")
    if elliott >= 0.8:
        parts.append("تصحيح فيبوناتشي مثالي 📐")
    elif elliott >= 0.6:
        parts.append("تصحيح صحي")
    if drop < -0.07:
        parts.append(f"تراجع {drop*100:.1f}%")
    elif abs(drop) < 0.02:
        parts.append("مستقر")
    return "، ".join(parts)

# ─── حساب درجة الثقة ─────────────────────────────────────────────
def confidence_score(prox, wyckoff, elliott):
    prox_score = max(0, 1 - (prox / 0.10))
    total = (prox_score * 0.60) + (wyckoff * 0.25) + (elliott * 0.15)
    return round(total * 100)

def trading_levels(hist, wyckoff, elliott, close):
    closes=hist["Close"].values; lows=hist["Low"].values
    strategy="wyckoff" if wyckoff>=0.6 else "elliott" if elliott>=0.6 else "classical"
    entry=round(float(close),4)
    if strategy=="wyckoff":
        stop=round(float(np.min(lows[-10:]))*0.97,4)
        t1=round(entry*1.15,4)
        t2=round(entry*1.25,4)
    elif strategy=="elliott":
        peak_idx=int(np.argmax(closes[-60:]))
        peak_val=float(closes[-60:][peak_idx])
        base_val=float(np.min(closes[-60:][:max(peak_idx,1)]))
        wave=peak_val-base_val
        stop=round(float(np.min(lows[-60:]))*0.98,4)
        t1=round(entry+wave*0.382,4)
        t2=round(entry+wave*0.618,4)
    else:
        stop=round(float(np.min(lows[-20:]))*0.98,4)
        t1=round(entry*1.10,4)
        t2=round(entry*1.20,4)
    return {"strategy_type":strategy,"entry":entry,"stop_loss":stop,"target1":t1,"target2":t2}

# ─── فحص سهم واحد ────────────────────────────────────────────────
def screen_stock(symbol, market="us"):
    tv_symbol = f"{symbol}.SR" if market == "sa" else symbol
    try:
        t    = yf.Ticker(tv_symbol)
        hist = t.history(period=f"{PERIOD_DAYS}d", interval="1d")

        if hist.empty or len(hist) < 20:
            print(f"  ⚠️  {symbol}: لا بيانات")
            return None

        if market == "us":
            avg_vol = hist["Volume"].tail(20).mean()
            if avg_vol < MIN_VOLUME:
                print(f"  ❌ {symbol}: حجم منخفض")
                return None

        score = proximity_score(hist)
        if score is None or score > MAX_SCORE:
            print(f"  ❌ {symbol}: سكور {score}")
            return None

        drop = hourly_drop(t)
        if drop < HOURLY_DROP:
            print(f"  ❌ {symbol}: تراجع ساعي {drop*100:.1f}%")
            return None

        wyckoff     = float(wyckoff_score(hist))
        elliott     = float(elliott_score(hist))
        consolidating, cons_reason = is_consolidating(hist)
        confidence  = int(confidence_score(score, wyckoff, elliott))
        note        = build_note(score, drop, wyckoff, elliott, consolidating, cons_reason)
        candles     = build_candles(hist)
        close       = round(float(hist["Close"].iloc[-1]), 4)
        levels      = trading_levels(hist, wyckoff, elliott, close)

        print(f"  ✅ {symbol}: سكور={score:.3f} | ثقة={confidence}% | Wyckoff={wyckoff} | Elliott={elliott} | ارتكاز={'✓' if consolidating else '✗'}")

        return {
            "score":         float(score),
            "confidence":    confidence,
            "wyckoff":       wyckoff,
            "elliott":       elliott,
            "consolidating": consolidating,
            "consReason":    cons_reason,
            "drop72h":       float(drop),
            "close":         close,
            "note":          note,
            "candles":       candles,
            "date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "strategy_type": levels["strategy_type"],
            "entry":         levels["entry"],
            "stop_loss":     levels["stop_loss"],
            "target1":       levels["target1"],
            "target2":       levels["target2"],
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

    top      = sorted(results.items(), key=lambda x: -x[1]["confidence"])[:TOP_N]
    top_cons = sorted(consolidating.items(), key=lambda x: -x[1]["confidence"])[:TOP_N]

    print(f"\n✅ فرص: {len(top)} | ارتكاز: {len(top_cons)}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── حفظ الفرص ──
    meta_ref = db.collection("meta").document(collection_us)
    meta_doc = meta_ref.get()
    version  = (meta_doc.to_dict() or {}).get("version", 0) + 1

    col_ref = db.collection(collection_us)
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
    meta_cons     = db.collection("meta").document(collection_cons)
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
