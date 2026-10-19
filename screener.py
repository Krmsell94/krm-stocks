"""
KRM Stocks Screener - v2
ثلاث طبقات: كلاسيكي + Wyckoff + Elliott
+ فرز الارتكاز (نطاق ضيق عند القاع)
يشتغل كل يوم تلقائياً عبر GitHub Actions
"""
 
import json
import os
from datetime import datetime, timezone, timedelta
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
# أُزيلت رموز ردّ عليها Yahoo بأنها ملغاة الإدراج:
# LILM, NKLA, FSR, ARVL, ASTR, VORB, SFET
US_WATCHLIST = [
    "EVLV", "PHAT", "SOUN", "ABAT", "LZ",
    "LFMD", "BW", "MARA", "RIOT", "CLSK",
    "CIFR", "WULF", "HIVE", "BTBT", "BSRT",
    "ACHR", "JOBY", "EOSE", "BLNK",
    "CHPT", "HYLN", 
    "SPCE", "RKLB", "MNTS", "ASTS",
    "LUNR", "RDW", "FLNC",
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
    """(close - period_low) / (period_high - period_low)
 
    تنبيه: بيانات السوق السعودي تأتي أحياناً بصفوف فيها NaN، فكانت
    النتيجة nan. وnan لا يساوي None ولا يزيد على أي رقم، فكان يمرّ
    من فلتر الفرز بلا رفض، ويُخرج ثقة 0% لكل سهم سعودي. لذلك ننظّف
    الصفوف الناقصة أولاً ونرفض أي نتيجة غير منتهية.
    """
    h = hist.dropna(subset=["High", "Low", "Close"])
    if h.empty or len(h) < 20:
        return None
    period_high = float(h["High"].max())
    period_low  = float(h["Low"].min())
    if not np.isfinite(period_high) or not np.isfinite(period_low):
        return None
    if period_high == period_low:
        return None
    close = float(h["Close"].iloc[-1])
    score = (close - period_low) / (period_high - period_low)
    if not np.isfinite(score):
        return None
    return round(score, 4)
 
def hourly_drop(ticker_obj):
    """تراجع السعر خلال آخر 4 أيام، مقيساً من أعلى إغلاق ساعي.
 
    كان يقيس من أعلى High لحظي، فذيل شمعة واحد — وهو شائع في الأسهم
    الصغيرة ضعيفة السيولة — يُنتج "تراجعاً" يصل 10% أو أكثر لسهم لم
    يتحرك فعلياً، فيُرفض بلا سبب. الإغلاق هو السعر الذي تراضى عليه
    السوق، فالقياس منه يصف الهبوط الحقيقي لا ضجيج الذيول.
    """
    try:
        h = ticker_obj.history(period="4d", interval="1h")
        if h.empty:
            return 0.0
        closes = h["Close"].dropna()
        if closes.empty:
            return 0.0
        high_72 = float(closes.max())
        last    = float(closes.iloc[-1])
        if not np.isfinite(high_72) or not np.isfinite(last) or high_72 <= 0:
            return 0.0
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
 
# ─── طبقة 3: Elliott Wave (مبسّط) ───────────────────────────────
def _zigzag(values, pct=0.08):
    """نقاط انعطاف السعر: كل انعكاس بنسبة pct أو أكثر يُسجّل محوراً.
    هذه هي الطريقة المتعارف عليها لتقسيم الحركة إلى موجات."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 3:
        return []
    piv = [(0, vals[0])]
    ei, ev = 0, vals[0]      # الطرف الحالي (أعلى قمة أو أدنى قاع منذ آخر محور)
    trend = 0                # 0 غير محدد، 1 صاعد، -1 هابط
    for i in range(1, n):
        v = vals[i]
        if trend == 0:
            if v >= vals[0] * (1 + pct):
                trend, ei, ev = 1, i, v
            elif v <= vals[0] * (1 - pct):
                trend, ei, ev = -1, i, v
            elif v > ev:
                ei, ev = i, v
        elif trend == 1:
            if v > ev:
                ei, ev = i, v
            elif v <= ev * (1 - pct):
                piv.append((ei, ev))
                trend, ei, ev = -1, i, v
        else:
            if v < ev:
                ei, ev = i, v
            elif v >= ev * (1 + pct):
                piv.append((ei, ev))
                trend, ei, ev = 1, i, v
    if piv[-1][0] != ei:
        piv.append((ei, ev))
    return piv
 
 
def elliott_score(hist):
    """إليوت في سياق البحث عن القاع.
 
    الصيغة القديمة كانت تقيس عمق التصحيح كنسبة فيبوناتشي من موجة
    صاعدة سابقة، وتشترط أن يكون بين 38% و79%. لكن كل سهم يمرّ فلترنا
    هو أصلاً على بعد 10% من أدنى 220 يوماً، أي أنه هبط تحت قاعدة
    الموجة نفسها، فتصير النسبة أكبر من 1 وتُرجع الدالة صفراً — وهذا
    ما ظهر في السجل: إليوت = 0.0 لستين رمزاً بلا استثناء، فكان وزنه
    (0.15) مهدوراً وسقف الثقة 85%.
 
    البديل يقيس ما يفيد فعلاً عند القاع: هل اكتمل التصحيح بنيوياً؟
      • ساقان هابطتان على الأقل (A و C) = بنية تصحيح لا بداية انهيار
      • الساق الأخيرة أضعف من الأولى = استنفاد البائعين
      • تباعد موجب: قاع أدنى في السعر مع RSI أعلى
    """
    h = hist.dropna(subset=["High", "Low", "Close"])
    if len(h) < 60:
        return 0.0
    closes = h["Close"].values
    peak_idx = int(np.argmax(closes))
    if len(closes) - 1 - peak_idx < 20:
        return 0.0                       # القمة حديثة جداً، لا هبوط نحلّله
    peak = float(closes[peak_idx])
    cur  = float(closes[-1])
    if not np.isfinite(peak) or not np.isfinite(cur) or peak <= 0:
        return 0.0
    if cur >= peak * 0.90:
        return 0.0                       # لم يصحّح بما يكفي ليكون قاعاً
 
    piv = _zigzag(closes[peak_idx:], pct=0.08)
    if len(piv) < 3:
        return 0.0
    legs = [(v1 - v2) / v1 for (_, v1), (_, v2) in zip(piv, piv[1:])
            if v2 < v1 and v1 > 0]
    if len(legs) < 2:
        return 0.0                       # ساق واحدة = انهيار مستمر لا تصحيح مكتمل
 
    score = 0.4
    if legs[-1] < legs[0] * 0.9:
        score += 0.3                     # استنفاد: الساق الأخيرة أقصر من الأولى
 
    rsi = rsi_series(h["Close"])
    if len(rsi) == len(closes) and len(closes) >= 50:
        tail = closes[-30:]
        last_low = int(np.argmin(tail)) + len(closes) - 30
        prev     = closes[:len(closes) - 30]
        if len(prev) >= 15:
            prev_low = int(np.argmin(prev))
            r1, r2 = float(rsi.iloc[last_low]), float(rsi.iloc[prev_low])
            if (np.isfinite(r1) and np.isfinite(r2)
                    and closes[last_low] < closes[prev_low] and r1 > r2 + 2):
                score += 0.3             # تباعد موجب
 
    return round(min(score, 1.0), 2)
 
 
# ═══════════════════════════════════════════════════════════════
# شروط سهم الارتكاز — مطابقة لدليل طريقة فيصل (ص 64-66، 73)
# المعادلة: ضغط قوي + دعم واضح + RSI 23-27 + تحسن MACD/فوليوم
#           + شمعة إيجابية + دخول قريب من الدعم
# ═══════════════════════════════════════════════════════════════
RSI_MIN, RSI_MAX = 23, 27    # ص65: منطقة الضغط المفضلة عند فيصل
RSI_WATCH_MAX    = 35        # ص65: فوق ~40 يعني الارتداد حصل أصلاً
 
def rsi_series(close, period=14):
    """RSI بتنعيم Wilder."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)
 
def macd_parts(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig
 
def macd_is_improving(close):
    """ص66: تقاطع إيجابي أو هيستوجرام يتحسن.
 
    ملاحظة مهمة: في سهم ينزل بتسارع يرتفع الهيستوجرام حسابياً بحكم
    تقارب المتوسطات، وهذا ليس تحسن زخم. لذلك نشترط مع صعود
    الهيستوجرام أن يكون السعر نفسه قد ارتد، وإلا فهو تحسن وهمي.
    """
    macd, sig, h = macd_parts(close)
    if len(h) < 4:
        return False, None
    rising     = bool(h.iloc[-1] > h.iloc[-2] > h.iloc[-3])
    price_up   = bool(close.iloc[-1] > close.iloc[-3])
    cross      = bool(macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2])
    return bool(cross or (rising and price_up)), round(float(h.iloc[-1]), 4)
 
def ma_state(hist):
    """ص64: السعر تحت 20/30/50 = ضغط، ثم يبدأ يستعيدها = تحسن."""
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    ma30 = close.rolling(30).mean()
    ma50 = close.rolling(50).mean()
    def last(s):
        v = s.iloc[-1]
        return None if np.isnan(v) else round(float(v), 4)
    vals = {"ma20": last(ma20), "ma30": last(ma30), "ma50": last(ma50)}
    c = float(close.iloc[-1])
 
    under = False
    if vals["ma20"] is not None and len(close) >= 20:
        below = (close.iloc[-20:] < ma20.iloc[-20:]).sum()
        under = bool(below >= 10)          # قضى نصف الشهر تحت المتوسط = ضغط
 
    recovering = False
    if vals["ma20"] and len(close) >= 6 and not np.isnan(ma20.iloc[-6]):
        gap_now  = (c - vals["ma20"]) / vals["ma20"]
        gap_prev = (float(close.iloc[-6]) - float(ma20.iloc[-6])) / float(ma20.iloc[-6])
        recovering = bool(gap_now > gap_prev and gap_now > -0.12)
    return under, recovering, vals
 
def bullish_candle_signal(hist):
    """ص66: Hammer / Engulfing / ذيل سفلي / رفض كسر."""
    if len(hist) < 2:
        return False, ""
    o = float(hist["Open"].iloc[-1]);  c = float(hist["Close"].iloc[-1])
    h = float(hist["High"].iloc[-1]);  l = float(hist["Low"].iloc[-1])
    po = float(hist["Open"].iloc[-2]); pc = float(hist["Close"].iloc[-2])
    rng = h - l
    if rng <= 0:
        return False, ""
    body  = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    if body > 0 and lower >= 2 * body and upper <= body:
        return True, "مطرقة"
    if lower / rng >= 0.5 and c > o:
        return True, "ذيل سفلي"
    if pc < po and c > o and c >= po and o <= pc:
        return True, "ابتلاع صاعد"
    return False, ""
 
def find_support(hist, lookback=60, tol=0.03, min_gap=3):
    """ص66: قاع/منطقة طلب يتم اختبارها.
 
    لا يصح أخذ أدنى سعر في الفترة كدعم، لسببين ظهرا في الاختبار:
    أولاً، في سهم ينزل باستمرار يكون أدنى سعر هو آخر شمعة، فيُفبرك
    دعم من سقوط حر. وثانياً، سحب السيولة نفسه يصنع قاعاً أدنى
    فيُلغي القاعدة الحقيقية فوقه. لذلك نبحث عن قيعان محورية
    اختُبرت أكثر من مرة في أوقات متباعدة.
    """
    lows = hist["Low"].values[-lookback:]
    if len(lows) < 25:
        return None, 0
 
    pivots = []
    for i in range(2, len(lows) - 2):
        window = lows[i - 2:i + 3]
        # ارتداد حقيقي: أدنى نقطة في النافذة، وأقل بوضوح من أعلاها.
        # الشرط الثاني يمنع احتساب النطاق المسطّح قاعاً محورياً.
        if lows[i] == window.min() and lows[i] < window.max() * 0.995:
            pivots.append(float(lows[i]))
    if not pivots:
        return None, 0
 
    cur = float(hist["Close"].iloc[-1])
    best = None
    for lvl in sorted(set(pivots)):
        if lvl <= 0 or lvl > cur * 1.03:
            continue      # مستوى فوق السعر الحالي مقاومة لا دعم
        hits, last = 0, -99
        for j, lo in enumerate(lows):
            # اللمسة قاع دخل نطاق المستوى، لا أي قاع أدنى منه. كان
            # الشرط (lo <= lvl*(1+tol)) بلا حدّ أدنى، فكل سعر تحت
            # المستوى يُحتسب لمسة، فيفوز أعلى محور بعدد لمسات وهمي
            # ويُرجع مستوى فوق السعر، فيسقط has_support لكل سهم —
            # وهذا سبب خروج صفر مرشح ارتكاز في السوقين.
            if lvl * (1 - tol) <= lo <= lvl * (1 + tol) and j - last >= min_gap:
                hits += 1
                last = j
        if hits < 2:
            continue
        # الأكثر اختباراً، وعند التساوي الأقرب للسعر لأنه الدعم العامل
        key = (hits, -abs(cur - lvl))
        if best is None or key > best[0]:
            best = (key, lvl, hits)
    if best is None:
        return None, 0
    return round(best[1], 4), best[2]
 
def failed_breakdown(hist, support, bars=10):
    """ص65/73: سحب سيولة أو فشل كسر الدعم ثم استرداد."""
    if not support:
        return False
    lows   = hist["Low"].values[-bars:]
    closes = hist["Close"].values[-bars:]
    for lo, cl in zip(lows, closes):
        if lo < support * 0.995 and cl > support:
            return True
    return False
 
def arkitaz_conditions(hist):
    """يرجع حالة كل شرط من شروط فيصل + هل السهم جاهز للدخول."""
    close = hist["Close"]
    # ص65: "ضغط سهم الارتكاز قبل الانفجار يجب أن يكون في منطقة 23-27"
    # الضغط يُقاس في مرحلة القاع، لا بعد الارتداد. لذلك نأخذ أدنى
    # قراءة خلال آخر 15 جلسة للضغط، والقراءة الحالية للتأخر.
    rsi_all = rsi_series(close)
    rsi_val = round(float(rsi_all.iloc[-1]), 2)
    rsi_low = round(float(rsi_all.iloc[-15:].min()), 2)
    macd_ok, macd_h   = macd_is_improving(close)
    under_ma, ma_rec, ma_vals = ma_state(hist)
    candle_ok, candle_name    = bullish_candle_signal(hist)
    sup, touches = find_support(hist)
    c = float(close.iloc[-1])
 
    has_sup = bool(sup and touches >= 2 and sup * 0.97 <= c <= sup * 1.25)
    swept   = failed_breakdown(hist, sup)
    # ثبات حقيقي: ثلاث جلسات فوق الدعم بهامش، وليس مجرد كونه فوق أدنى سعر
    stable  = bool(sup and len(close) >= 3
                   and all(x > sup * 1.01 for x in close.values[-3:]))
 
    # ص65: تحسن الفوليوم مع الارتداد، لا مع الهبوط
    vol = hist["Volume"].values
    vol_ok = bool(len(vol) >= 20 and c > float(close.iloc[-2])
                  and vol[-1] > float(np.mean(vol[-20:])) * 0.8)
 
    # ص65: التحذير من RSI المرتفع (حول 40 = الارتداد حصل)، لا من الأعمق.
    # لذلك شرط الضغط هو بلوغ 27 فأقل، و23-27 هي المنطقة المثالية فقط.
    rsi_pressure = bool(rsi_low <= RSI_MAX)
    rsi_ideal    = bool(RSI_MIN <= rsi_low <= RSI_MAX)
    rsi_ok       = bool(rsi_pressure and rsi_val <= 40)
    rsi_watch    = bool(rsi_low <= RSI_WATCH_MAX)
 
    # مرشّح لتبويب الارتكاز: مضغوط + عنده دعم حقيقي
    candidate = bool(has_sup and rsi_watch and under_ma)
 
    # ص73: "لا يدخل لأن السهم قد يصعد، بل لأن الهبوط فشل والشروط اكتملت"
    ready = bool(has_sup and rsi_ok and macd_ok and candle_ok
                 and (swept or stable) and ma_rec)
 
    missing = []
    if not has_sup:              missing.append("دعم واضح")
    if not rsi_ok:
        if rsi_val > 40:         missing.append(f"RSI {rsi_val} — الارتداد حصل")
        else:                    missing.append(f"لا ضغط كافٍ (أدنى RSI {rsi_low})")
    if not macd_ok:              missing.append("MACD لم يتحسن")
    if not candle_ok:            missing.append("لا شمعة إيجابية")
    if not (swept or stable):    missing.append("الهبوط لم يفشل بعد")
    if not ma_rec:               missing.append("لم يستعد المتوسطات")
 
    # ثقة الارتكاز مستقلة عن ثقة الفرص: تلك وزنها 0.60 للقرب من أدنى
    # 220 يوماً، وسهم الارتكاز قد يكون أعلى من قاعه بكثير وهو مع ذلك
    # على دعم ممتاز. فنقيسها بشروط فيصل نفسها.
    ark_conf = round(100 * (
        (0.25 if has_sup   else 0) + (0.20 if rsi_ok   else 0) +
        (0.15 if macd_ok   else 0) + (0.15 if (swept or stable) else 0) +
        (0.10 if candle_ok else 0) + (0.10 if ma_rec   else 0) +
        (0.05 if vol_ok    else 0)))
 
    return {
        "ark_confidence": ark_conf,
        "rsi":            rsi_val,
        "rsi_low15":      rsi_low,
        "rsi_ideal":      rsi_ideal,
        "macd_improving": macd_ok,
        "macd_hist":      macd_h,
        "ma_recovering":  ma_rec,
        "under_ma":       under_ma,
        "ma20":           ma_vals["ma20"],
        "ma30":           ma_vals["ma30"],
        "ma50":           ma_vals["ma50"],
        "bullish_candle": candle_ok,
        "candle_type":    candle_name,
        "has_support":    has_sup,
        "support_level":  sup,
        "support_touches": touches,
        "failed_breakdown": swept,
        "holding_support":  stable,
        "volume_ok":      vol_ok,
        "ark_candidate":  candidate,
        "ark_ready":      ready,
        "ark_missing":    missing,
    }
 
# ─── فرز الارتكاز ────────────────────────────────────────────────
def is_consolidating(hist):
    if len(hist) < 30:
        return False, ""
    closes  = hist["Close"].values
    volumes = hist["Volume"].values
    recent_high  = hist["High"].values[-7:].max()
    recent_low   = hist["Low"].values[-7:].min()
    range_pct    = (recent_high - recent_low) / closes[-7:].mean()
    tight        = range_pct < 0.08
    avg_vol_20   = volumes[-20:].mean()
    recent_vol7  = volumes[-7:].mean()
    quiet        = recent_vol7 < avg_vol_20 * 0.85
    month_high   = hist["High"].values[-22:].max()
    month_low    = hist["Low"].values[-22:].min()
    month_range  = month_high - month_low
    if month_range == 0:
        return False, ""
    pos_in_range = (closes[-1] - month_low) / month_range
    near_support = pos_in_range < 0.25
    month_perf   = (closes[-1] - closes[-22]) / closes[-22] if len(closes) >= 22 else 0
    monthly_down = month_perf <= 0.03
    reasons = []
    if tight:        reasons.append("نطاق ضيق")
    if quiet:        reasons.append("حجم هادئ")
    if near_support: reasons.append("عند الدعم")
    if monthly_down: reasons.append("أداء محايد")
    is_cons = bool(tight and quiet and near_support)
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
 
# ─── حساب درجة الثقة الكلية ──────────────────────────────────────
def confidence_score(prox, wyckoff, elliott):
    if prox is None or not np.isfinite(prox):
        return 0
    prox_score = max(0.0, 1.0 - (prox / 0.10))
    total = (prox_score * 0.60) + (wyckoff * 0.25) + (elliott * 0.15)
    return round(total * 100)
 
# ─── جاهزية الفرص ────────────────────────────────────────────────
# معادلة الثقة تمنح القرب من القاع وزن 0.60، وهذا يرفع درجة السهم
# الساقط لأنه يصنع قيعاناً جديدة كل يوم. فالقرب من القاع ليس دليل
# ارتداد. لذلك الجاهزية تشترط تأكيداً بنيوياً من وايكوف أو إليوت،
# وتوقفاً فعلياً للنزول، ولا تكتفي بالدرجة وشمعة خضراء واحدة.
def foras_readiness(hist, wyckoff, elliott, confidence):
    close, low, op, vol = hist["Close"], hist["Low"], hist["Open"], hist["Volume"]
    c = float(close.iloc[-1])
    missing = []
 
    conf_ok = bool(confidence >= 60)
    if not conf_ok:
        missing.append(f"ثقة {confidence}%")
 
    # لا يكفي القرب من القاع — لا بد أن تؤكد إحدى الطبقتين البنيويتين
    structure_ok = bool(wyckoff >= 0.6 or elliott >= 0.6)
    if not structure_ok:
        missing.append("لا تأكيد وايكوف/إليوت")
 
    # لم يعمّق قاعه: قيعان آخر ثلاث جلسات ليست أدنى مما قبلها
    base_ok = False
    if len(low) >= 6:
        base_ok = bool(float(low.iloc[-3:].min()) >= float(low.iloc[-6:-3].min()) * 0.98)
    if not base_ok:
        missing.append("ما زال يعمّق القاع")
 
    # ارتد فعلاً خلال ثلاث جلسات، لا شمعة خضراء واحدة داخل نزول
    turned_up = bool(len(close) >= 4 and c > float(close.iloc[-4]))
    if not turned_up:
        missing.append("لم يرتد بعد")
 
    green = bool(c > float(op.iloc[-1]))
    if not green:
        missing.append("شمعة حمراء")
 
    vol_ok = False
    if len(vol) >= 20:
        vol_ok = bool(float(vol.iloc[-1]) >= float(vol.iloc[-20:].mean()) * 0.8)
    if not vol_ok:
        missing.append("فوليوم ضعيف")
 
    ready = bool(conf_ok and structure_ok and base_ok and turned_up and green and vol_ok)
    return ready, missing
 
# ─── مستويات التداول ─────────────────────────────────────────────
def trading_levels(hist, wyckoff, elliott, close):
    closes = hist["Close"].values
    lows   = hist["Low"].values
    if wyckoff >= 0.6:
        strategy = "wyckoff"
    elif elliott >= 0.6:
        strategy = "elliott"
    else:
        strategy = "classical"
    entry = round(float(close), 4)
    if strategy == "wyckoff":
        stop = round(float(np.min(lows[-10:])) * 0.97, 4)
        t1   = round(entry * 1.15, 4)
        t2   = round(entry * 1.25, 4)
    elif strategy == "elliott":
        peak_idx = int(np.argmax(closes[-60:]))
        peak_val = float(closes[-60:][peak_idx])
        base_val = float(np.min(closes[-60:][:max(peak_idx, 1)]))
        wave     = peak_val - base_val
        stop     = round(float(np.min(lows[-60:])) * 0.98, 4)
        t1       = round(entry + wave * 0.382, 4)
        t2       = round(entry + wave * 0.618, 4)
    else:
        stop = round(float(np.min(lows[-20:])) * 0.98, 4)
        t1   = round(entry * 1.10, 4)
        t2   = round(entry * 1.20, 4)
    return {
        "strategy_type": strategy,
        "entry":         entry,
        "stop_loss":     stop,
        "target1":       t1,
        "target2":       t2,
    }
 
# ─── فحص سهم واحد ────────────────────────────────────────────────
def float_shares(ticker_obj):
    """الأسهم الحرة. المتصفح لا يستطيع جلبها (يحجب Yahoo الاستدعاء
    من نطاق آخر)، فنجلبها هنا ونكتبها جاهزة للعرض."""
    try:
        info = ticker_obj.info or {}
    except Exception:
        return None
    val = info.get("floatShares") or info.get("sharesOutstanding")
    if not val or val <= 0:
        return None
    val = float(val)
    if val >= 1e9:  return f"{val/1e9:.2f}B"
    if val >= 1e6:  return f"{val/1e6:.1f}M"
    return f"{val:,.0f}"
 
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
 
        # فلاتر الفرص (القرب من القاع + التراجع الساعي) تخصّ تبويب الفرص
        # وحده. الارتكاز عند فيصل سهم على دعم حقيقي، وقد يكون أعلى من
        # أدنى 220 يوماً بكثير. فلو رفضنا السهم هنا لما وصل الارتكاز
        # أبداً — وهذا ما حدث: صفر مرشح ارتكاز في السوقين، لأن التبويب
        # كان يُغذّى من ناجي فلاتر الفرص فقط. فنفصل: الرفض يُعلّم السهم
        # بأنه ليس فرصة، لا أن يُحذف من الفرز كله.
        score = proximity_score(hist)
        prox_ok = bool(score is not None and np.isfinite(score) and score <= MAX_SCORE)
 
        if prox_ok:
            drop = hourly_drop(t)
            drop_ok = bool(drop >= HOURLY_DROP)
            if not drop_ok:
                print(f"  ⤵️  {symbol}: خارج الفرص — تراجع ساعي {drop*100:.1f}%")
        else:
            drop, drop_ok = 0.0, True      # لا داعي لنداء شبكة إضافي
            print(f"  ⤵️  {symbol}: خارج الفرص — سكور {score}")
 
        foras_pass = bool(prox_ok and drop_ok)
        if score is None or not np.isfinite(score):
            score = 1.0                     # بعيد عن القاع: لا يحتسب في الفرص
 
        wyckoff = wyckoff_score(hist)
        elliott = elliott_score(hist)
        consolidating, cons_reason = is_consolidating(hist)
        confidence = confidence_score(score, wyckoff, elliott)
        ark = arkitaz_conditions(hist)      # شروط فيصل (تبويب الارتكاز)
        flt = float_shares(t)
        foras_ready, foras_missing = foras_readiness(hist, wyckoff, elliott, confidence)
        if not foras_pass:
            foras_ready = False
            foras_missing = ["خارج فلاتر الفرص"] + foras_missing
 
        note    = build_note(score, drop, wyckoff, elliott, consolidating, cons_reason)
        candles = build_candles(hist)
        close   = round(float(hist["Close"].iloc[-1]), 4)
        levels  = trading_levels(hist, wyckoff, elliott, close)
 
        news = []
        try:
            raw_news = t.news or []
            for n in raw_news[:5]:
                news.append({
                    "title":     n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link":      n.get("link", ""),
                    "time":      n.get("providerPublishTime", 0),
                })
        except:
            news = []
 
        ark_mark = '🟢 جاهز' if ark["ark_ready"] else ('🟡 مرشح' if ark["ark_candidate"] else '✗')
        foras_mark = ('🟢 جاهز' if foras_ready
                      else ('🟡 متابعة' if foras_pass else '✗ خارج'))
        print(f"  ✅ {symbol}: سكور={score:.3f} | ثقة={confidence}% | Wyckoff={wyckoff} | "
              f"Elliott={elliott} | RSI={ark['rsi']} | فرص={foras_mark} "
              f"| ارتكاز={ark_mark} ({ark['ark_confidence']}%)")
        if foras_pass and not foras_ready:
            print(f"      فرص ناقص: {'، '.join(foras_missing)}")
        if ark["ark_candidate"] and not ark["ark_ready"]:
            print(f"      ارتكاز ناقص: {'، '.join(ark['ark_missing'])}")
 
        return {
            "score":         float(score),
            "confidence":    int(confidence),
            "wyckoff":       float(wyckoff),
            "elliott":       float(elliott),
            "consolidating": bool(consolidating),
            "consReason":    cons_reason,
            "drop72h":       float(drop),
            "close":         float(close),
            "note":          note,
            "candles":       candles,
            "date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "strategy_type": levels["strategy_type"],
            "entry":         float(levels["entry"]),
            "stop_loss":     float(levels["stop_loss"]),
            "target1":       float(levels["target1"]),
            "target2":       float(levels["target2"]),
            "news":          news,
            "float_shares":  flt,
            "foras_pass":    foras_pass,
            "foras_ready":   foras_ready,
            "foras_missing": foras_missing,
            **ark,
        }
 
    except Exception as e:
        print(f"  ⚠️  {symbol}: خطأ {e}")
        return None
 
# ─── إغلاق الصفقات الورقية آلياً ─────────────────────────────────
# بدون هذا تبقى الخاسرة مفتوحة إلى الأبد بينما تُغلق الرابحة لجني
# الربح، فتميل نسبة النجاح لصالح الرابحين ببنية القياس لا بالأداء.
MAX_HOLD_DAYS = 30   # صفقة لم تبلغ هدفاً ولا وقفاً خلال شهر تُغلق بسعر السوق
 
def close_open_paper_trades():
    print(f"\n{'='*40}\n🧾 الصفقات الورقية\n{'='*40}")
    riyadh = timezone(timedelta(hours=3))
    col = db.collection("paper_trades")
    try:
        docs = list(col.stream())
    except Exception as e:
        print(f"  ⚠️  تعذّرت القراءة: {e}")
        return
 
    opened = [d for d in docs if (d.to_dict() or {}).get("status") == "open"]
    if not opened:
        print("  لا توجد صفقات مفتوحة")
        return
 
    closed_n = 0
    for d in opened:
        t = d.to_dict()
        sym    = t.get("symbol")
        market = t.get("market", "us")
        entry  = t.get("entry")
        stop   = t.get("stop_loss")
        tgt    = t.get("target1")
        since  = t.get("registeredAt", "")[:10]
        if not (sym and entry and stop and tgt and since):
            continue
 
        yf_sym = f"{sym}.SR" if market == "sa" else sym
        try:
            hist = yf.Ticker(yf_sym).history(start=since, interval="1d")
        except Exception as e:
            print(f"  ⚠️  {sym}: {e}")
            continue
        if hist.empty:
            continue
 
        close_price = close_date = outcome = None
        for idx, row in hist.iterrows():
            lo, hi = float(row["Low"]), float(row["High"])
            # لو لمس الوقف والهدف في نفس الجلسة نفترض الوقف أولاً.
            # الشمعة اليومية لا تكشف الترتيب، والافتراض المتحفظ أصدق.
            if lo <= stop:
                close_price, outcome = stop, "stop"
            elif hi >= tgt:
                close_price, outcome = tgt, "target"
            if close_price is not None:
                close_date = idx.strftime("%Y-%m-%d")
                break
 
        if close_price is None:
            held = (datetime.now(timezone.utc).date()
                    - datetime.strptime(since, "%Y-%m-%d").date()).days
            if held >= MAX_HOLD_DAYS:
                close_price = round(float(hist["Close"].iloc[-1]), 4)
                close_date  = hist.index[-1].strftime("%Y-%m-%d")
                outcome     = "expired"
 
        if close_price is None:
            continue
 
        pnl = round((close_price - entry) / entry * 100, 2)
        col.document(d.id).update({
            "status":      "closed",
            "close_price": round(float(close_price), 4),
            "close_date":  close_date,
            "outcome":     outcome,
            "pnl_pct":     pnl,
            "closedBy":    "auto",
            "closedAt":    datetime.now(riyadh).strftime("%Y-%m-%d %H:%M"),
        })
        closed_n += 1
        mark = {"target": "🎯 هدف", "stop": "🛑 وقف", "expired": "⌛ انتهت المدة"}[outcome]
        print(f"  {mark} {sym}: دخول {entry} → {close_price} ({pnl:+.2f}%)")
 
    print(f"\n  أُغلقت {closed_n} من {len(opened)} مفتوحة")
 
# ─── مؤشرات السوق (لشريط حالة السوق في الواجهة) ─────────────────
# الواجهة على GitHub Pages استضافة ساكنة بلا MCP، فلا تستطيع جلب
# هذي البيانات بنفسها. نجلبها هنا ونكتبها في Firestore.
MARKET_INDICES = {
    "us": [("IWM",  "AMEX:IWM",  "الأسهم الصغيرة"),
           ("IWC",  "AMEX:IWC",  "المتناهية الصغر"),
           ("^VIX", "CBOE:VIX",  "مؤشر الخوف"),
           ("SPY",  "AMEX:SPY",  "السوق كله")],
    # ^TASI.SR لم يرجع بيانات في التشغيل، فنجرّب عدة رموز بالترتيب
    "sa": [(["^TASI", "^TASI.SR", "TASI.SR"], "TADAWUL:TASI", "المؤشر العام"),
           (["BZ=F", "BNO"],                  "ICEEUR:BRN1!", "خام برنت")],
}
 
def index_snapshot(yf_symbol):
    """إغلاق + تغير% + متوسط 50 و200 + RSI14 لمؤشر واحد."""
    t = yf.Ticker(yf_symbol)
    hist = t.history(period="300d", interval="1d")
    if hist.empty or len(hist) < 30:
        return None
    close = hist["Close"]
    c    = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    out = {
        "close":  round(c, 4),
        "change": round((c - prev) / prev * 100, 2) if prev else None,
        "SMA50":  round(float(close.rolling(50).mean().iloc[-1]), 4)  if len(close) >= 50  else None,
        "SMA200": round(float(close.rolling(200).mean().iloc[-1]), 4) if len(close) >= 200 else None,
        "RSI":    round(float(rsi_series(close).iloc[-1]), 1),
    }
    return out
 
def run_market_indices():
    print(f"\n{'='*40}\n📈 مؤشرات السوق\n{'='*40}")
    # بتوقيت الرياض لا UTC: يُعرض بجانب ساعة الرياض الحية في الواجهة
    riyadh = timezone(timedelta(hours=3))
    today = datetime.now(riyadh).strftime("%Y-%m-%d %H:%M")
    for mk, rows in MARKET_INDICES.items():
        data = {}
        for yf_sym, key, label in rows:
            # قد يكون الرمز قائمة بدائل (تاسي مثلاً)، فنجرّبها بالترتيب
            candidates = yf_sym if isinstance(yf_sym, (list, tuple)) else [yf_sym]
            snap, used = None, None
            for cand in candidates:
                try:
                    snap = index_snapshot(cand)
                except Exception as e:
                    print(f"  ⚠️  {cand}: {e}")
                    snap = None
                if snap:
                    used = cand
                    break
                print(f"  ⚠️  {cand}: لا بيانات")
            if not snap:
                print(f"  ❌ {key}: فشلت كل البدائل {candidates}")
                continue
            if used != candidates[0]:
                print(f"  ↩️  {key}: استُخدم البديل {used}")
            snap["label"] = label
            snap["ticker"] = key.split(":")[1]
            data[key] = snap
            print(f"  ✅ {key}: {snap['close']} ({snap['change']}%) RSI={snap['RSI']}")
        if not data:
            print(f"  ❌ {mk}: لم تُجلب أي مؤشرات — أُبقي القراءة السابقة")
            continue
        db.collection("market").document(mk).set({"data": data, "updated": today})
        print(f"  💾 market/{mk} ← {len(data)} مؤشر")
 
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
            # الفرص: من مرّ فلاتر القرب من القاع والتراجع الساعي
            if data["foras_pass"]:
                results[sym] = data
            # الارتكاز: شروط فيصل وحدها، مستقلة عن فلاتر الفرص
            if data["ark_candidate"]:
                consolidating[sym] = data
 
    top = sorted(results.items(), key=lambda x: -x[1]["confidence"])[:TOP_N]
    # الجاهز للدخول أولاً، ثم الأقرب لاكتمال الشروط
    top_cons = sorted(consolidating.items(),
                      key=lambda x: (not x[1]["ark_ready"], len(x[1]["ark_missing"]),
                                     -x[1]["confidence"]))[:TOP_N]
 
    n_ready = sum(1 for _, d in top_cons if d["ark_ready"])
    print(f"\n✅ فرص: {len(top)} | ارتكاز: {len(top_cons)} (جاهز للدخول: {n_ready})")
 
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
 
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
 
    try:
        run_market_indices()
    except Exception as e:
        print(f"⚠️  مؤشرات السوق فشلت ولم تؤثر على الفرز: {e}")
 
    try:
        close_open_paper_trades()
    except Exception as e:
        print(f"⚠️  إغلاق الصفقات فشل ولم يؤثر على الفرز: {e}")
 
    print("\n" + "="*40)
    print("📊 ملخص")
    print("="*40)
    print(f"🇺🇸 فرص: {[s for s,_ in us_top]}")
    print(f"🇺🇸 ارتكاز: {[s for s,_ in us_cons]}")
    print(f"🇸🇦 فرص: {[s for s,_ in sa_top]}")
    print(f"🇸🇦 ارتكاز: {[s for s,_ in sa_cons]}")
    print("\n✅ انتهى الفرز")
 
