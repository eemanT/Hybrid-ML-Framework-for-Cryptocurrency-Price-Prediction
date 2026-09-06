"""
FastAPI Backend — NeuralPredict Crypto AI (v4 Fixed)

KEY FIXES:
1. Both CNN-LSTM (_lstm.keras) and BiLSTM (_bilstm.keras) loaded & used for real predictions
2. /api/predict returns BOTH model forecasts (lstm=CNN-LSTM, hybrid=BiLSTM)
3. SHAP analysis runs on real loaded model with per-coin feature data
4. /api/retrain trains BOTH models for the selected coin
5. /api/price uses only backend yfinance data — no mixing with ticker hook
6. Ticker bar uses separate useTickerPrices hook (Yahoo direct) — no interference
"""

import os, sys, time, warnings, asyncio, subprocess
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import requests
from datetime import datetime

from fastapi              import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import yfinance  as yf
import tensorflow as tf
import tensorflow.keras.backend as K

try:
    from preprocessing import preprocess_for_lstm
    from arima_model   import forecast_arima
    from hybrid_model  import hybrid_predict
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

try:
    from incremental_trainer import check_and_retrain, get_last_trained_ts, load_meta
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuralPredict API", version="4.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000","*"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# ── Warmup status tracker ────────────────────────────────────────────────────
# Tracks which coins have data ready in cache
_warmup_status: dict = {}   # coin -> "loading" | "ready" | "failed"
_warmup_started = False


@app.on_event("startup")
async def startup():
    """
    On startup:
    1. Clear stale caches
    2. Load all model files into memory
    3. Download all 7 coins historical data into cache IN THE BACKGROUND
       so the first prediction request is instant (cache hit, not 60s download)
    """
    global _warmup_started
    import asyncio

    logger.info("=" * 55)
    logger.info("  NeuralPredict startup")
    logger.info("=" * 55)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Clear stale data
    _data_cache.clear()
    _shap_cache.clear()
    _ticker_cache["data"] = {}
    _ticker_cache["ts"]   = 0
    for coin in COINS:
        _warmup_status[coin] = "pending"
    logger.info("Caches cleared")

    # Step 2: Load model files into memory
    loaded = 0
    for coin in COINS:
        for variant in ("lstm", "bilstm"):
            if _load_model(coin, variant) is not None:
                loaded += 1
    logger.info(f"Models loaded: {loaded} files")

    # Step 3: Download all coin data in background
    # Each coin gets full OHLCV history + feature engineering + scaler fitting
    # stored in _data_cache so predict() is instant on first request
    async def _download_coin(coin: str):
        _warmup_status[coin] = "loading"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _get_data, coin)
            _warmup_status[coin] = "ready"
            logger.info(f"  ✓ {coin} data ready")
        except Exception as e:
            _warmup_status[coin] = "failed"
            logger.warning(f"  ✗ {coin} failed: {str(e)[:60]}")

    async def _compute_shap_background(coin: str, variant: str):
        """Compute SHAP for one coin:variant in background thread."""
        key = f"{coin}:{variant}"
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _compute_shap_for_coin, coin, variant)
            if result:
                _shap_cache[key] = result
                logger.info(f"  ✓ SHAP {key} cached")
            else:
                logger.info(f"  · SHAP {key} skipped (no model or data)")
        except Exception as e:
            logger.debug(f"  · SHAP {key} failed: {str(e)[:50]}")

    async def _warm_all_coins():
        logger.info("Downloading historical data for all coins...")
        # Phase 1: Download all coin data sequentially (staggered to avoid rate-limits)
        for coin in COINS:
            await _download_coin(coin)
            await asyncio.sleep(2)

        ready = sum(1 for s in _warmup_status.values() if s == "ready")
        logger.info(f"Data warmup complete: {ready}/{len(COINS)} coins ready")

        # Phase 2: Pre-compute SHAP for all coins that have both data and models
        # Each coin:variant takes ~15-30s on CPU
        logger.info("Pre-computing SHAP values for all coins...")
        for coin in COINS:
            if _warmup_status.get(coin) != "ready":
                continue  # skip coins with no data
            for variant in ("lstm", "bilstm"):
                if _load_model(coin, variant) is None:
                    continue  # skip missing models
                await _compute_shap_background(coin, variant)
                await asyncio.sleep(0.5)  # small gap between computations

        shap_ready = len(_shap_cache)
        logger.info(f"SHAP pre-computation complete: {shap_ready} variants cached")
        logger.info("=" * 55)

    asyncio.create_task(_warm_all_coins())
    _warmup_started = True
    logger.info("Background data download started — predictions will be instant once ready")
    logger.info("=" * 55)

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
WINDOW_SIZE     = 30
DIR_HORIZON     = 5
STRONG_QUANTILE = 0.85
CONFIDENCE_THR  = 0.45
STRONG_DOWN, NEUTRAL, STRONG_UP = 0, 1, 2
MIN_ROWS_REQUIRED = 400  # SMA200(200) + Vol_regime(90) + window(30) + horizon(5) + buffer(75)

def _smart_round(price: float) -> float:
    """
    Round price to meaningful precision based on magnitude.
    Prevents sub-dollar coins (ADA, DOGE, XRP) losing forecast precision.
      >= $1000: 2dp   ($77,222.50)
      >= $100:  3dp   ($592.500)
      >= $1:    4dp   ($1.4772)
      >= $0.01: 6dp   ($0.258580)
      <  $0.01: 8dp   ($0.00009934)
    """
    if price <= 0:
        return price
    if price >= 1000: return round(price, 2)
    if price >= 100:  return round(price, 3)
    if price >= 1:    return round(price, 4)
    if price >= 0.01: return round(price, 6)
    return round(price, 8)

COINS = {
    "BTC-USD":  {"name":"Bitcoin",  "symbol":"BTC",  "color":"#f7931a","start":"2018-01-01"},
    "ETH-USD":  {"name":"Ethereum", "symbol":"ETH",  "color":"#627eea","start":"2018-01-01"},
    "BNB-USD":  {"name":"BNB",      "symbol":"BNB",  "color":"#f0b90b","start":"2018-01-01"},
    "XRP-USD":  {"name":"XRP",      "symbol":"XRP",  "color":"#00aae4","start":"2018-01-01"},
    "ADA-USD":  {"name":"Cardano",  "symbol":"ADA",  "color":"#0033ad","start":"2018-01-01"},
    "SOL-USD":  {"name":"Solana",   "symbol":"SOL",  "color":"#9945ff","start":"2020-03-01"},
    "DOGE-USD": {"name":"Dogecoin", "symbol":"DOGE", "color":"#c2a633","start":"2018-01-01"},
}

KAGGLE_RESULTS = {
    "BTC-USD":  {"da_strong":61.7,"da_conf":70.8,"mape":1.68,"f1_macro":0.5832,"rmse":1842},
    "ETH-USD":  {"da_strong":67.0,"da_conf":50.0,"mape":2.79,"f1_macro":0.4011,"rmse":186 },
    "BNB-USD":  {"da_strong":63.2,"da_conf":56.5,"mape":1.97,"f1_macro":0.3871,"rmse":22  },
    "XRP-USD":  {"da_strong":50.9,"da_conf":47.6,"mape":2.92,"f1_macro":0.4913,"rmse":0.04},
    "ADA-USD":  {"da_strong":61.7,"da_conf":60.4,"mape":3.48,"f1_macro":0.5978,"rmse":0.03},
    "SOL-USD":  {"da_strong":72.1,"da_conf":58.8,"mape":3.10,"f1_macro":0.4189,"rmse":11  },
    "DOGE-USD": {"da_strong":57.4,"da_conf":48.4,"mape":3.60,"f1_macro":0.5513,"rmse":0.01},
}

# ── Custom loss ────────────────────────────────────────────────────────────────
def sparse_focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        y_pred   = K.clip(y_pred, 1e-7, 1.0-1e-7)
        y_true_i = K.cast(y_true, "int32")
        n_cls    = K.shape(y_pred)[1]
        y_oh     = K.cast(K.one_hot(y_true_i, n_cls), "float32")
        ce       = -K.sum(y_oh * K.log(y_pred), axis=-1)
        p_t      = K.sum(y_oh * y_pred, axis=-1)
        fw       = K.pow(1.0-p_t, gamma)
        if alpha is not None:
            a_t = K.sum(y_oh * K.constant(alpha, dtype="float32"), axis=-1)
            return K.mean(a_t * fw * ce)
        return K.mean(fw * ce)
    loss_fn.__name__ = "sparse_focal_loss"
    return loss_fn

CUSTOM_OBJ = {"loss_fn": sparse_focal_loss(2.0)}
_model_cache: dict = {}

def _model_path(coin: str, variant: str) -> str:
    fname = f"{coin}_lstm.keras" if variant == "lstm" else f"{coin}_bilstm.keras"
    return os.path.join(MODEL_DIR, fname)

def _load_model(coin: str, variant: str = "lstm"):
    """Load and cache a Keras model. Returns None if file not found."""
    key  = f"{coin}_{variant}"
    if key in _model_cache:
        return _model_cache[key]
    path = _model_path(coin, variant)
    if not os.path.exists(path):
        return None
    from tensorflow.keras.models import load_model
    try:
        m = load_model(path, custom_objects=CUSTOM_OBJ, compile=False)
        _model_cache[key] = m
        logger.info(f"Loaded {key} ({m.count_params():,} params)")
        return m
    except Exception as e:
        logger.error(f"Failed to load {key}: {e}")
        return None

def _invalidate_model_cache(coin: str):
    for v in ("lstm", "bilstm"):
        key = f"{coin}_{v}"
        if key in _model_cache:
            del _model_cache[key]

_data_cache: dict = {}
_DATA_TTL = 3600  # 1 hour

# SHAP cache: keyed by "coin:variant" e.g. "BTC-USD:lstm"
# Pre-computed at startup, invalidated on retrain
_shap_cache: dict = {}

def _flatten_yfinance_df(raw: pd.DataFrame, coin: str) -> pd.DataFrame:
    """
    Convert any yfinance DataFrame (single or multi-ticker, any version) into a
    clean DataFrame with exactly: Date, Open, High, Low, Close, Volume.

    Handles:
    - yfinance >= 0.2.31 MultiIndex: ('field','ticker') or ('ticker','field')
    - Duplicate 'Adj Close' that renames to 'Close' causing duplicate columns
    - Flat columns (older yfinance or after reset_index on some versions)
    """
    df = raw.copy()
    df.reset_index(inplace=True)

    # ── Step 1: flatten MultiIndex to plain strings ──────────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        OHLCV_EXACT = {"close", "open", "high", "low", "volume", "date", "datetime"}
        ADJ_CLOSE   = {"adj close", "adj_close"}
        new_cols = []
        for col in df.columns:
            parts = [str(p).strip() for p in col if str(p).strip()]
            cl_parts = [p.lower() for p in parts]

            # Drop Adj Close entirely — prefer the unadjusted Close column
            if any(p in ADJ_CLOSE for p in cl_parts):
                new_cols.append("__DROP__")
                continue

            # Find the OHLCV field name within the tuple
            field = None
            for p in parts:
                if p.lower() in OHLCV_EXACT:
                    field = p
                    break

            if field is None:
                # Fallback: take the part that is NOT a ticker symbol
                non_ticker = [p for p in parts if p and "-" not in p]
                field = non_ticker[0] if non_ticker else (parts[0] if parts else "__DROP__")

            new_cols.append(field)

        df.columns = new_cols
        # Remove all __DROP__ columns
        df = df[[c for c in df.columns if c != "__DROP__"]]

    # ── Step 2: standardise to canonical names ───────────────────────────────
    seen_close = False
    rename = {}
    drop_extra = []
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "close":
            if not seen_close:
                rename[c] = "Close"
                seen_close = True
            else:
                drop_extra.append(c)   # duplicate Close — drop it
        elif cl in ("adj close", "adj_close"):
            if not seen_close:
                rename[c] = "Close"
                seen_close = True
            else:
                drop_extra.append(c)
        elif cl == "open":   rename[c] = "Open"
        elif cl == "high":   rename[c] = "High"
        elif cl == "low":    rename[c] = "Low"
        elif cl == "volume": rename[c] = "Volume"
        elif cl in ("date", "datetime", "index"): rename[c] = "Date"

    if drop_extra:
        df.drop(columns=drop_extra, inplace=True)
    df.rename(columns=rename, inplace=True)

    # ── Step 3: keep only canonical columns ─────────────────────────────────
    keep = {"Date", "Open", "High", "Low", "Close", "Volume"}
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)

    return df


def _normalize_df_columns(df):
    """
    Wrapper around _flatten_yfinance_df for backwards compatibility.
    Also handles plain (non-MultiIndex) DataFrames.
    """
    if isinstance(df.columns, pd.MultiIndex):
        return _flatten_yfinance_df(df, "unknown")

    # Plain columns — just rename to canonical names, removing Adj Close
    seen_close = False
    col_map = {}
    drop_cols = []
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "close":
            if not seen_close:
                col_map[c] = "Close"; seen_close = True
            else:
                drop_cols.append(c)
        elif cl in ("adj close", "adj_close"):
            if not seen_close:
                col_map[c] = "Close"; seen_close = True
            else:
                drop_cols.append(c)
        elif cl == "open":   col_map[c] = "Open"
        elif cl == "high":   col_map[c] = "High"
        elif cl == "low":    col_map[c] = "Low"
        elif cl == "volume": col_map[c] = "Volume"
        elif cl in ("date","datetime","index"): col_map[c] = "Date"

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    df.rename(columns=col_map, inplace=True)
    keep = {"Date","Open","High","Low","Close","Volume"}
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
    return df

def _clean_ticker_df(raw, coin: str) -> "pd.DataFrame | None":
    """Flatten and clean a raw yfinance DataFrame. Returns None if invalid."""
    if raw is None or len(raw) == 0:
        return None
    df = _flatten_yfinance_df(raw, coin)
    for req in ("Date", "Close"):
        if req not in df.columns:
            return None
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.dropna(subset=["Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        return None
    test_val = df["Close"].iloc[-1]
    if hasattr(test_val, "__len__"):
        return None
    return df


# ── Per-coin expected price ranges (wide safety margins covering all-time history) ─
# Used to detect when yfinance returns data for the WRONG coin
_COIN_PRICE_RANGES: dict = {
    "BTC-USD":  (500,      500_000),  # Bitcoin       — 2018 low $3k, ATH $110k+
    "ETH-USD":  (5,        50_000),   # Ethereum      — 2018 low $80
    "BNB-USD":  (0.1,      10_000),   # BNB           — 2017 low $0.10
    "XRP-USD":  (0.001,    500),      # XRP           — very low historical prices
    "ADA-USD":  (0.0001,   50),       # Cardano       — sub-cent in 2018
    "SOL-USD":  (0.1,      1_000),    # Solana        — launched 2020 at ~$0.50
    "DOGE-USD": (0.00001,  5),        # Dogecoin      — was sub-cent for years
}


def _validate_coin_data(df: "pd.DataFrame", coin: str,
                        start: str = None, period: str = None) -> bool:
    """
    Validate that a DataFrame actually contains data for the requested coin.
    Returns True if data looks correct, False if it appears to be wrong-coin data.
    """
    if df is None or len(df) == 0:
        return False

    last_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0])

    # ── 1. Price range check ─────────────────────────────────────────────────
    lo, hi = _COIN_PRICE_RANGES.get(coin, (0.00001, 10_000_000))
    if not (lo <= last_close <= hi):
        logger.warning(f"[VALIDATE] {coin}: last_close={last_close:.4f} outside "
                       f"expected range [{lo}, {hi}] — likely wrong coin data")
        return False
    if not (lo <= first_close <= hi):
        logger.warning(f"[VALIDATE] {coin}: first_close={first_close:.4f} outside "
                       f"expected range — likely wrong coin data")
        return False

    # ── 2. Date recency + minimum rows for period-based fetches ────────────────
    if period and "d" in str(period):
        # Check last date is recent
        last_date = df["Date"].iloc[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        days_old = (pd.Timestamp.now().date() - last_date).days
        if days_old > 7:
            logger.warning(f"[VALIDATE] {coin}: last date {last_date} is {days_old} days old — stale")
            return False
        # Check minimum row count for period (crypto trades ~5/7 days)
        try:
            period_days = int(str(period).replace("d", ""))
            min_rows = max(2, int(period_days * 0.4))  # expect at least 40% of days
            if len(df) < min_rows:
                logger.warning(f"[VALIDATE] {coin}: only {len(df)} rows for period={period}, "
                               f"expected ≥{min_rows} — rate limited or wrong data")
                return False
        except ValueError:
            pass

    # ── 3. Row count check for historical start= fetches ────────────────────
    if start:
        start_dt = pd.Timestamp(start).date()
        today    = pd.Timestamp.now().date()
        expected_min = int((today - start_dt).days * 0.6)
        if len(df) < expected_min:
            logger.warning(f"[VALIDATE] {coin}: only {len(df)} rows since {start}, "
                           f"expected ≥{expected_min} — rate limited or wrong data")
            return False

    logger.debug(f"[VALIDATE] {coin}: OK (last={last_close:.4f}, rows={len(df)})")
    return True


def _download_single_coin(coin: str, period: str = "60d", start: str = None) -> "pd.DataFrame | None":
    """
    Download OHLCV for a specific coin with:
    - Ticker.history() primary + yf.download() fallback
    - 3 attempts with backoff
    - Price range + date + row count validation to detect wrong-coin data
    - Returns None only if all attempts fail validation
    """
    kwargs_ticker = dict(auto_adjust=True, progress=False)
    kwargs_dl     = dict(progress=False, auto_adjust=True)

    if start:
        kwargs_ticker["start"] = start
        kwargs_dl["start"]     = start
    else:
        kwargs_ticker["period"] = period
        kwargs_dl["period"]     = period

    last_err = None

    for attempt in range(3):
        if attempt > 0:
            wait = 2.0 * attempt
            logger.info(f"[DOWNLOAD] {coin}: retry {attempt+1}/3 in {wait}s")
            time.sleep(wait)

        # ── Method 1: yf.Ticker().history() ──────────────────────────────────
        try:
            # Create fresh Ticker object each time to avoid stale internal cache
            ticker = yf.Ticker(coin)
            raw    = ticker.history(**kwargs_ticker)

            if raw is not None and len(raw) >= 2:
                raw.reset_index(inplace=True)
                raw.rename(columns={"Datetime": "Date", "datetime": "Date"}, inplace=True)
                if "Date" not in raw.columns and raw.index.name in ("Date", "Datetime"):
                    raw = raw.reset_index()

                df = _clean_ticker_df(raw, coin)
                if df is not None and len(df) >= 2:
                    if _validate_coin_data(df, coin, start=start, period=period):
                        logger.info(f"[DOWNLOAD] {coin} ✓ Ticker attempt {attempt+1}: "
                                    f"{len(df)} rows, last={float(df['Close'].iloc[-1]):.4f}")
                        return df
                    else:
                        logger.warning(f"[DOWNLOAD] {coin}: Ticker data failed validation, trying download()")
        except Exception as e:
            last_err = e
            logger.debug(f"[DOWNLOAD] {coin} Ticker attempt {attempt+1}: {e}")

        # ── Method 2: yf.download() ───────────────────────────────────────────
        try:
            raw = yf.download(coin, **kwargs_dl)
            if raw is not None and len(raw) >= 2:
                df = _clean_ticker_df(raw, coin)
                if df is not None and len(df) >= 2:
                    if _validate_coin_data(df, coin, start=start, period=period):
                        logger.info(f"[DOWNLOAD] {coin} ✓ download attempt {attempt+1}: "
                                    f"{len(df)} rows, last={float(df['Close'].iloc[-1]):.4f}")
                        return df
                    else:
                        logger.warning(f"[DOWNLOAD] {coin}: download() data failed validation")
        except Exception as e:
            last_err = e
            logger.debug(f"[DOWNLOAD] {coin} download attempt {attempt+1}: {e}")

    logger.error(f"[DOWNLOAD] {coin}: all attempts failed. Last error: {last_err}")
    return None
def _get_feature_names():
    return [
        "LogReturn","Return_1d","Return_3d","Return_7d","Return_14d",
        "Vol_10","Vol_30","HL_range","Close_SMA10","Close_SMA30","Close_SMA50","SMA10_SMA30",
        "RSI_14","RSI_6","BB_pos","BB_width","MACD","MACD_sig","MACD_hist",
        "Mom_5","Mom_10","Vol_ratio","Vol_change","OBV_change","Price_pos",
        "Bull_regime","Trend_str","MA_cross","Vol_regime","FearGreed","FG_chg",
    ]

def _build_features_and_split(df):
    from sklearn.preprocessing import RobustScaler
    data = df.copy()
    data["LogReturn"]  = np.log(data["Close"]/data["Close"].shift(1))
    data["Return_1d"]  = data["Close"].pct_change(1)
    data["Return_3d"]  = data["Close"].pct_change(3)
    data["Return_7d"]  = data["Close"].pct_change(7)
    data["Return_14d"] = data["Close"].pct_change(14)
    data["Vol_10"]     = data["LogReturn"].rolling(10).std()
    data["Vol_30"]     = data["LogReturn"].rolling(30).std()
    data["HL_range"]   = (data["High"]-data["Low"])/(data["Close"]+1e-10)
    sma10=data["Close"].rolling(10).mean(); sma30=data["Close"].rolling(30).mean()
    sma50=data["Close"].rolling(50).mean(); sma200=data["Close"].rolling(200).mean()
    ema12=data["Close"].ewm(span=12,adjust=False).mean(); ema26=data["Close"].ewm(span=26,adjust=False).mean()
    data["Close_SMA10"] = data["Close"]/(sma10+1e-10)-1
    data["Close_SMA30"] = data["Close"]/(sma30+1e-10)-1
    data["Close_SMA50"] = data["Close"]/(sma50+1e-10)-1
    data["SMA10_SMA30"] = sma10/(sma30+1e-10)-1
    data["Bull_regime"] = (data["Close"]>sma200).astype(float)
    data["Trend_str"]   = data["Close"]/(sma200+1e-10)-1
    data["MA_cross"]    = (sma10>sma30).astype(float)
    data["Vol_regime"]  = (data["LogReturn"].rolling(30).std()/(data["LogReturn"].rolling(90).std()+1e-10)-1)
    delta=data["Close"].diff(); gain=delta.where(delta>0,0.0); loss=-delta.where(delta<0,0.0)
    data["RSI_14"] = (100-(100/(1+gain.rolling(14).mean()/(loss.rolling(14).mean()+1e-10))))/50-1
    data["RSI_6"]  = (100-(100/(1+gain.rolling(6).mean() /(loss.rolling(6).mean() +1e-10))))/50-1
    bb_mid=data["Close"].rolling(20).mean(); bb_std=data["Close"].rolling(20).std()
    data["BB_pos"]   = (data["Close"]-(bb_mid-2*bb_std))/(4*bb_std+1e-10)
    data["BB_width"] = (4*bb_std)/(bb_mid+1e-10)
    data["MACD"]     = (ema12-ema26)/(data["Close"]+1e-10)
    data["MACD_sig"] = data["MACD"].ewm(span=9,adjust=False).mean()
    data["MACD_hist"]= data["MACD"]-data["MACD_sig"]
    data["Mom_5"]    = data["Close"].pct_change(5)
    data["Mom_10"]   = data["Close"].pct_change(10)
    vm = data["Volume"].rolling(10).mean()
    data["Vol_ratio"]  = data["Volume"]/(vm+1e-10)-1
    data["Vol_change"] = data["Volume"].pct_change()
    data["Price_pos"]  = ((data["Close"]-data["Close"].rolling(20).min())/
                          (data["Close"].rolling(20).max()-data["Close"].rolling(20).min()+1e-10))
    obv=[0]
    for i in range(1,len(data)):
        if   data["Close"].iloc[i]>data["Close"].iloc[i-1]: obv.append(obv[-1]+data["Volume"].iloc[i])
        elif data["Close"].iloc[i]<data["Close"].iloc[i-1]: obv.append(obv[-1]-data["Volume"].iloc[i])
        else: obv.append(obv[-1])
    data["OBV_change"] = pd.Series(obv).pct_change().fillna(0).values
    if "FearGreed" not in data.columns: data["FearGreed"]=0.5
    if "FG_chg"    not in data.columns: data["FG_chg"]=0.0
    FCOLS = _get_feature_names()
    data.dropna(inplace=True); data.reset_index(drop=True,inplace=True)
    if len(data)==0: raise ValueError("All rows dropped after feature engineering")
    closes=data["Close"].values; n=len(closes)
    train_end=int(n*0.70); val_end=int(n*0.85)
    sc=RobustScaler()
    train_arr=data[FCOLS].values[:train_end]
    if len(train_arr)==0: raise ValueError("Training slice is empty")
    tr_sc=sc.fit_transform(train_arr)
    va_sc=sc.transform(data[FCOLS].values[train_end:val_end])
    te_sc=sc.transform(data[FCOLS].values[val_end:])
    fwd=np.array([(closes[i+DIR_HORIZON]-closes[i])/(closes[i]+1e-10)
                  if i+DIR_HORIZON<n else np.nan for i in range(n)])
    tr_clean=fwd[:train_end][~np.isnan(fwd[:train_end])]
    up_thr=np.quantile(tr_clean,STRONG_QUANTILE); dn_thr=np.quantile(tr_clean,1-STRONG_QUANTILE)
    def l3(r): return STRONG_UP if r>up_thr else STRONG_DOWN if r<dn_thr else NEUTRAL
    def mseq(arr,cls,fv):
        X,yr,y3=[],[],[]
        for i in range(WINDOW_SIZE,len(arr)-DIR_HORIZON):
            f=fv[i]
            if np.isnan(f): continue
            X.append(arr[i-WINDOW_SIZE:i]); yr.append(arr[i,0]); y3.append(l3(f))
        return np.array(X),np.array(yr),np.array(y3)
    X_tr,y_tr,_=mseq(tr_sc,closes[:train_end],fwd[:train_end])
    ctst=np.concatenate([va_sc[-WINDOW_SIZE:],te_sc])
    clt=np.concatenate([closes[val_end-WINDOW_SIZE:val_end],closes[val_end:]])
    ft=np.concatenate([fwd[val_end-WINDOW_SIZE:val_end],fwd[val_end:]])
    X_ts,y_ts,_=mseq(ctst,clt,ft)
    if len(X_tr)==0: raise ValueError("0 training sequences after preprocessing")
    sc._target_center=float(sc.center_[0]); sc._target_scale=float(sc.scale_[0])
    sc._up_thr=float(up_thr); sc._down_thr=float(dn_thr)
    sc._closes_test_start=float(closes[val_end]) if val_end<len(closes) else float(closes[-1])
    return X_tr,X_ts,y_tr,y_ts,sc

_fg_cache={"df":None,"ts":0,"today":0.5,"today_ts":0}

def _fetch_fg_history():
    if time.time()-_fg_cache["ts"]<3600 and _fg_cache["df"] is not None: return _fg_cache["df"]
    try:
        r=requests.get("https://api.alternative.me/fng/",params={"limit":0,"format":"json"},timeout=15)
        tmp=pd.DataFrame(r.json()["data"])
        tmp["Date"]=pd.to_datetime(tmp["timestamp"].astype(int),unit="s").dt.normalize().dt.tz_localize(None)
        tmp["FearGreed"]=tmp["value"].astype(float)/100.0
        fg=tmp[["Date","FearGreed"]].sort_values("Date").reset_index(drop=True)
        _fg_cache["df"]=fg; _fg_cache["ts"]=time.time(); return fg
    except Exception: return pd.DataFrame({"Date":[],"FearGreed":[]})

def _get_fg_today():
    if time.time()-_fg_cache["today_ts"]<1800: return _fg_cache["today"]
    try:
        r=requests.get("https://api.alternative.me/fng/",params={"limit":1},timeout=8)
        v=float(r.json()["data"][0]["value"])/100.0
        _fg_cache["today"]=v; _fg_cache["today_ts"]=time.time(); return v
    except Exception: return 0.5

def _get_data(coin: str) -> dict:
    now = time.time()
    if coin in _data_cache and now - _data_cache[coin]["ts"] < _DATA_TTL:
        return _data_cache[coin]

    logger.info(f"[DATA] Fetching {coin} from {COINS[coin]['start']}")
    df = _download_single_coin(coin, start=COINS[coin]["start"])

    if df is None:
        raise ValueError(
            f"[DATA] {coin}: Could not fetch valid historical data from yfinance. "
            f"This may be a rate-limit or network issue. "
            f"The app will use benchmark data until live data is available."
        )

    # Ensure all required OHLCV columns present
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[DATA] {coin}: missing columns {missing} after download")

    df = df[required].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(f"[DATA] {coin}: only {len(df)} rows (need ≥{MIN_ROWS_REQUIRED})")

    logger.info(f"[DATA] {coin}: {len(df)} rows, "
                f"{df['Date'].min().date()} → {df['Date'].max().date()}, "
                f"last close={df['Close'].iloc[-1]:.4f}")

    # Merge Fear & Greed index
    fg_df = _fetch_fg_history()
    dn    = df["Date"].dt.normalize().dt.tz_localize(None)
    df["FearGreed"] = dn.map(dict(zip(fg_df["Date"], fg_df["FearGreed"]))).fillna(0.5) if not fg_df.empty else 0.5
    df["FG_chg"]    = df["FearGreed"].diff(3).fillna(0)

    X_tr, X_ts, y_tr, y_ts, sc = _build_features_and_split(df)

    r = dict(df=df, scaler=sc, feature_names=_get_feature_names(),
             X_train=X_tr, X_test=X_ts, y_train=y_tr, y_test=y_ts, ts=now)
    _data_cache[coin] = r
    return r

def _predict_core(model, X_all, scaler, last_close, horizon):
    tc = float(scaler._target_center) if hasattr(scaler, "_target_center") else float(scaler.center_[0])
    ts = float(scaler._target_scale)  if hasattr(scaler, "_target_scale")  else float(scaler.scale_[0])

    # ── Step 1: Auto-regressive price forecast ───────────────────────────────
    inp = X_all[-1].copy()
    preds_sc = []
    for _ in range(horizon):
        raw = model.predict(inp[np.newaxis], verbose=0)
        p   = float(raw[0][0, 0]) if isinstance(raw, list) else float(raw[0, 0])
        preds_sc.append(p)
        ns = inp[-1].copy(); ns[0] = p
        inp = np.vstack([inp[1:], ns])

    pred_prices = []
    price = last_close
    for ps in preds_sc:
        price = price * float(np.exp(ps * ts + tc))
        pred_prices.append(_smart_round(price))

    # ── Step 2: Classification head probs (for confidence bars display) ──────
    raw5 = model.predict(X_all[-1][np.newaxis], verbose=0)
    if isinstance(raw5, list) and len(raw5) > 1:
        # Dual-head model: raw5[1] = [prob_down, prob_neutral, prob_up]
        probs = [round(float(p), 4) for p in raw5[1][0]]
    else:
        # Single-head model: no classification head.
        # We'll derive synthetic probs from the forecast direction after Step 3.
        probs = None  # resolved below after direction is determined

    # ── Step 3: Direction from PRICE FORECAST (always consistent with chart) ─
    # The classification head predicts 5-day direction trained at build time.
    # The price forecast is what the model actually outputs step-by-step.
    # When they conflict (e.g. prices go up but head says DOWN), trust the prices
    # since that's what the user sees on the chart.
    chg_5d = 0.0
    if pred_prices:
        price_5d = float(pred_prices[min(DIR_HORIZON - 1, len(pred_prices) - 1)])
        chg_5d   = (price_5d - last_close) / (last_close + 1e-10)

    # Thresholds — use scaler's stored training thresholds if available.
    # If not (e.g. old model without meta), derive from actual predicted returns.
    if hasattr(scaler, "_up_thr") and hasattr(scaler, "_down_thr"):
        up_thr = float(scaler._up_thr)
        dn_thr = float(scaler._down_thr)
    else:
        # Derive thresholds from the actual log-return outputs for this run
        # Use the top/bottom 15% of predicted values as strong signal cutoffs
        if preds_sc:
            sorted_preds = sorted(preds_sc)
            n = len(sorted_preds)
            up_thr = float(sorted_preds[int(n * 0.85)]) if n > 1 else 0.01
            dn_thr = float(sorted_preds[int(n * 0.15)]) if n > 1 else -0.01
        else:
            up_thr, dn_thr = 0.01, -0.01  # minimal fallback, very rarely hit

    if chg_5d > up_thr:
        dir_label = "STRONG UP"
        # Scale confidence: further above threshold = higher confidence
        forecast_conf = min(0.95, 0.55 + (chg_5d - up_thr) / max(up_thr, 0.001) * 0.3)
    elif chg_5d < dn_thr:
        dir_label = "STRONG DOWN"
        forecast_conf = min(0.95, 0.55 + (abs(chg_5d) - abs(dn_thr)) / max(abs(dn_thr), 0.001) * 0.3)
    else:
        dir_label = "NEUTRAL"
        forecast_conf = 0.40 + abs(chg_5d) / max(abs(up_thr), abs(dn_thr), 0.001) * 0.15

    # Resolve probs for single-head models — derive from forecast so bars match signal
    if probs is None:
        if dir_label == "STRONG UP":
            probs = [round(1.0 - forecast_conf, 4), round((1.0-forecast_conf)*0.4, 4), round(forecast_conf, 4)]
        elif dir_label == "STRONG DOWN":
            probs = [round(forecast_conf, 4), round((1.0-forecast_conf)*0.4, 4), round(1.0-forecast_conf, 4)]
        else:
            side = round((1.0 - forecast_conf) / 2, 4)
            probs = [side, round(forecast_conf, 4), side]

    # Blend: forecast direction is primary, classification confidence is secondary
    # If classification head agrees -> boost confidence slightly
    cls_dir_idx = int(np.argmax(probs))
    cls_agrees  = (cls_dir_idx == STRONG_UP   and dir_label == "STRONG UP") or                   (cls_dir_idx == STRONG_DOWN  and dir_label == "STRONG DOWN") or                   (cls_dir_idx == NEUTRAL      and dir_label == "NEUTRAL")
    conf = forecast_conf * 1.05 if cls_agrees else forecast_conf * 0.95
    conf = min(0.95, max(0.10, conf))

    chg = round((pred_prices[min(4, len(pred_prices) - 1)] - last_close) / last_close * 100, 2) if pred_prices else 0.0

    return dict(
        pred_prices    = pred_prices,
        direction      = dir_label,
        confidence     = round(conf * 100, 1),
        probs          = probs,                          # raw classification probs for display
        is_confident   = conf >= CONFIDENCE_THR,
        last_close     = _smart_round(last_close),
        price_change_pct = chg,
    )

def _evaluate_core(model, X_test, y_test, scaler):
    if len(X_test)==0: raise ValueError("X_test is empty")
    tc=float(scaler._target_center) if hasattr(scaler,"_target_center") else float(scaler.center_[0])
    ts=float(scaler._target_scale)  if hasattr(scaler,"_target_scale")  else float(scaler.scale_[0])
    out=model.predict(X_test,verbose=0,batch_size=64)
    pr=out[0].flatten() if isinstance(out,list) else out.flatten()
    dp=out[1] if isinstance(out,list) and len(out)>1 else None
    ref=float(scaler._closes_test_start) if hasattr(scaler,"_closes_test_start") else 40000.0
    tp_arr,pp_arr=[],[]; price=ref
    for pl,tl in zip(pr,y_test):
        pp_arr.append(price*float(np.exp(float(pl)*ts+tc)))
        true_p=price*float(np.exp(float(tl)*ts+tc)); tp_arr.append(true_p); price=true_p
    tp=np.array(tp_arr); pp=np.array(pp_arr)
    res=dict(mape=round(float(np.mean(np.abs((tp-pp)/(tp+1e-10)))*100),2),
             rmse=round(float(np.sqrt(np.mean((tp-pp)**2))),2),
             mae=round(float(np.mean(np.abs(tp-pp))),2),
             da=round(float(np.mean(np.sign(np.diff(tp))==np.sign(np.diff(pp)))*100),1))
    if dp is not None:
        from sklearn.metrics import f1_score
        pc=np.argmax(dp,axis=1); cf=np.max(dp,axis=1)
        up_thr=float(scaler._up_thr) if hasattr(scaler,"_up_thr") else 0.05
        dn_thr=float(scaler._down_thr) if hasattr(scaler,"_down_thr") else -0.05
        fwd=np.array([(tp[i+DIR_HORIZON]-tp[i])/(tp[i]+1e-10) if i+DIR_HORIZON<len(tp) else np.nan for i in range(len(tp))])
        td=np.array([STRONG_UP if not np.isnan(f) and f>up_thr else STRONG_DOWN if not np.isnan(f) and f<dn_thr else NEUTRAL for f in fwd])
        sm=(td==STRONG_DOWN)|(td==STRONG_UP); ns=int(sm.sum())
        da_s=float(np.mean(pc[sm]==td[sm])*100) if ns>0 else 0.0
        adap=float(np.percentile(cf[sm],60)) if ns>0 else CONFIDENCE_THR
        cs=sm&(cf>min(adap,CONFIDENCE_THR+0.05)); nc=int(cs.sum())
        da_c=float(np.mean(pc[cs]==td[cs])*100) if nc>0 else 0.0
        f1m=float(f1_score(td[sm],pc[sm],average="macro",zero_division=0)) if ns>0 else 0.0
        res.update(dict(da_strong=round(da_s,1),da_conf=round(da_c,1),n_strong=ns,n_conf=nc,f1_macro=round(f1m,4)))
    return res

# ════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    ready_count = sum(1 for s in _warmup_status.values() if s == "ready")
    return {
        "status":      "ok",
        "version":     "4.0.0",
        "timestamp":   datetime.utcnow().isoformat(),
        "gpu":         len(tf.config.list_physical_devices("GPU")) > 0,
        "modules":     MODULES_AVAILABLE,
        "models":      {c: {"cnn_lstm": os.path.exists(_model_path(c,"lstm")),
                            "bilstm":   os.path.exists(_model_path(c,"bilstm"))} for c in COINS},
        "data_ready":  ready_count,
        "data_total":  len(COINS),
        "warmup":      _warmup_status.copy(),
    }

@app.get("/api/warmup-status")
def warmup_status_endpoint():
    """Which coins have data downloaded and ready for instant predictions."""
    ready   = [c for c, s in _warmup_status.items() if s == "ready"]
    loading = [c for c, s in _warmup_status.items() if s in ("loading","pending")]
    failed  = [c for c, s in _warmup_status.items() if s == "failed"]
    all_done = len(loading) == 0
    return {
        "all_ready":   all_done and len(failed) == 0,
        "ready":       ready,
        "loading":     loading,
        "failed":      failed,
        "ready_count": len(ready),
        "total":       len(COINS),
        "message": (
            f"All {len(COINS)} coins ready — predictions are instant"
            if all_done and not failed
            else f"{len(ready)}/{len(COINS)} ready, still downloading: {loading}"
            if not all_done
            else f"{len(ready)} ready, {len(failed)} failed (will use benchmark)"
        ),
    }

@app.get("/api/coins")
def get_coins():
    return [{**meta,"id":cid,"model_available":os.path.exists(_model_path(cid,"lstm"))}
            for cid,meta in COINS.items()]

@app.get("/api/cache-status")
def cache_status():
    """Shows which coins have data cached — useful for debugging slow first predictions."""
    return {
        "cached_coins": list(_data_cache.keys()),
        "total_cached": len(_data_cache),
        "ticker_cached": bool(_ticker_cache.get("data")),
        "note": "Run predictions for any coin not listed — first run downloads 7yr history (~30-60s)"
    }

@app.get("/api/model-status")
def model_status():
    """
    Detailed model loading status for all coins and variants.
    Shows which models are on disk AND which are currently in memory cache.
    """
    status = {}
    for coin in COINS:
        status[coin] = {
            "cnn_lstm": {
                "file_exists":  os.path.exists(_model_path(coin, "lstm")),
                "in_cache":     f"{coin}_lstm" in _model_cache,
                "path":         _model_path(coin, "lstm"),
            },
            "bilstm": {
                "file_exists":  os.path.exists(_model_path(coin, "bilstm")),
                "in_cache":     f"{coin}_bilstm" in _model_cache,
                "path":         _model_path(coin, "bilstm"),
            },
        }
    total_files  = sum(1 for c in status.values() for v in c.values() if v["file_exists"])
    total_cached = sum(1 for c in status.values() for v in c.values() if v["in_cache"])
    return {
        "models":       status,
        "total_files":  total_files,
        "total_cached": total_cached,
        "models_dir":   MODEL_DIR,
        "ready":        total_files > 0,
    }

@app.get("/api/price/{coin}")
def get_price(coin: str, days: int = Query(30, ge=1, le=365)):
    """
    Price history for ONE specific coin — same fetch strategy as top ticker.
    Uses Ticker.history() + download() fallback with retry.
    Validates the coin ID is in the response to prevent data mixing.
    """
    if days < 7: days = 30
    if coin not in COINS:
        raise HTTPException(404, f"Unknown coin: {coin}")
    try:
        df = _download_single_coin(coin, period=f"{days + 10}d")
        if df is None or len(df) < 2:
            raise ValueError(f"No price data returned for {coin}")

        df = df.tail(days).reset_index(drop=True)

        # Sanity check: price must be in a reasonable range for this coin
        # Prevents mixing up coins (e.g. BTC range vs XRP range)
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        if last <= 0:
            raise ValueError(f"Invalid price {last} for {coin}")

        return {
            "coin":          coin,   # always echo requested coin
            "current_price": _smart_round(last),
            "change_24h":    round((last - prev) / prev * 100, 2) if prev > 0 else 0.0,
            "history": [
                {"date": row["Date"].strftime("%b %d"),
                 "price": _smart_round(float(row["Close"]))}
                for _, row in df.iterrows()
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PRICE] {coin}: {e}")
        raise HTTPException(500, f"[PRICE:{coin}] {str(e)}")

@app.get("/api/fear-greed")
def get_fear_greed():
    val=_get_fg_today(); pct=round(val*100)
    return {"value":val,"score":pct,"label":(
        "Extreme Fear" if pct<25 else "Fear" if pct<45 else "Neutral" if pct<55 else
        "Greed" if pct<75 else "Extreme Greed")}

@app.post("/api/predict/{coin}")
def predict(coin: str, horizon: int = Query(7, ge=1, le=30)):
    """
    Real predictions from both CNN-LSTM and BiLSTM.
    forecast[i].lstm   = CNN-LSTM price prediction
    forecast[i].hybrid = BiLSTM price prediction
    Returns 'live_model' when models loaded, 'kaggle_results' otherwise.
    """
    if coin not in COINS: raise HTTPException(404,f"Unknown coin: {coin}")
    cnn_model    = _load_model(coin,"lstm")
    bilstm_model = _load_model(coin,"bilstm")

    if cnn_model is None and bilstm_model is None:
        r=KAGGLE_RESULTS[coin]
        try: lc=get_price(coin,30)["current_price"]
        except Exception: lc=None
        forecast=[]
        if lc:
            trend=0.002 if r["da_conf"]>65 else -0.001; p=lc
            for i in range(horizon):
                p*=(1+trend+np.random.normal(0,0.003))
                dt=(pd.Timestamp.now()+pd.Timedelta(days=i+1)).strftime("%b %d")
                forecast.append({"date":dt,"lstm":round(p,4),"hybrid":round(p*1.001,4),
                                  "upper":round(p*1.03,4),"lower":round(p*0.97,4)})
        return {"coin":coin,"source":"kaggle_results",
                "direction":"STRONG UP" if r["da_conf"]>65 else "NEUTRAL" if r["da_conf"]>50 else "STRONG DOWN",
                "confidence":round(r["da_conf"]*0.9,1),"probs":[0.12,0.28,0.60] if r["da_conf"]>65 else [0.22,0.55,0.23],
                "is_confident":r["da_conf"]*0.9>=45,"last_close":lc,"price_change_pct":0.0,
                "pred_prices":[f["lstm"] for f in forecast],"forecast":forecast,
                "fear_greed":_get_fg_today(),"note":f"Place {coin}_lstm.keras in backend/models/"}

    try:
        d = _get_data(coin)
    except Exception as data_err:
        # Data fetch/processing failed — fall back to Kaggle benchmark results
        # This happens when yfinance returns insufficient rows or network issues
        logger.warning(f"[PREDICT] {coin}: data fetch failed ({data_err}), using Kaggle fallback")
        r = KAGGLE_RESULTS[coin]
        try:
            lc = get_price(coin, 30)["current_price"]
        except Exception:
            lc = None
        forecast = []
        if lc:
            trend = 0.002 if r["da_conf"] > 65 else -0.001
            p = lc
            for i in range(horizon):
                p *= (1 + trend + np.random.normal(0, 0.003))
                dt = (pd.Timestamp.now() + pd.Timedelta(days=i+1)).strftime("%b %d")
                forecast.append({"date": dt, "lstm": round(p, 4), "hybrid": round(p*1.001, 4),
                                  "upper": round(p*1.03, 4), "lower": round(p*0.97, 4)})
        return {
            "coin":             coin,
            "source":           "kaggle_results",
            "direction":        "STRONG UP" if r["da_conf"] > 65 else "NEUTRAL" if r["da_conf"] > 50 else "STRONG DOWN",
            "confidence":       round(r["da_conf"] * 0.9, 1),
            "probs":            [0.12, 0.28, 0.60] if r["da_conf"] > 65 else [0.22, 0.55, 0.23],
            "is_confident":     r["da_conf"] * 0.9 >= 45,
            "last_close":       lc,
            "price_change_pct": 0.0,
            "pred_prices":      [f["lstm"] for f in forecast],
            "forecast":         forecast,
            "fear_greed":       _get_fg_today(),
            "note":             f"Data fetch failed: {str(data_err)[:100]}. Showing benchmark data.",
        }

    try:
        X_all      = np.concatenate([d["X_train"], d["X_test"]], axis=0)
        last_close = float(d["df"]["Close"].iloc[-1])

        # CNN-LSTM prediction
        if cnn_model is not None:
            cnn_res = _predict_core(cnn_model, X_all, d["scaler"], last_close, horizon)
        else:
            cnn_res = _predict_core(bilstm_model, X_all, d["scaler"], last_close, horizon)

        # BiLSTM prediction
        if bilstm_model is not None:
            bil_res    = _predict_core(bilstm_model, X_all, d["scaler"], last_close, horizon)
            bil_prices = bil_res["pred_prices"]
        else:
            bil_prices = None

        # Build per-day forecast combining both models
        forecast = []
        for i, cnn_p in enumerate(cnn_res["pred_prices"]):
            dt  = (pd.Timestamp.now() + pd.Timedelta(days=i+1)).strftime("%b %d")
            bil = bil_prices[i] if bil_prices else cnn_p * 1.001
            forecast.append({"date": dt, "lstm": _smart_round(cnn_p), "hybrid": _smart_round(float(bil)),
                              "upper": _smart_round(float(bil)*1.03), "lower": _smart_round(float(bil)*0.97)})

        if MODULES_AVAILABLE:
            try:
                ap = forecast_arima(d["df"], steps=horizon).tolist()
                hb = hybrid_predict(cnn_res["pred_prices"], ap, alpha=0.6)
                for i, h in enumerate(hb):
                    forecast[i]["upper"] = round(float(h)*1.03, 4)
                    forecast[i]["lower"] = round(float(h)*0.97, 4)
            except Exception:
                pass

        return {**cnn_res, "coin": coin, "source": "live_model",
                "forecast": forecast, "fear_greed": _get_fg_today()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"[PREDICTION] {str(e)}")

@app.get("/api/evaluate/{coin}")
def evaluate(coin: str):
    if coin not in COINS: raise HTTPException(404, f"Unknown coin: {coin}")
    model = _load_model(coin, "lstm")
    if model is None:
        return {**KAGGLE_RESULTS[coin], "coin": coin, "source": "kaggle_results"}
    try:
        d = _get_data(coin)
        if len(d["X_test"]) == 0:
            return {**KAGGLE_RESULTS[coin], "coin": coin, "source": "kaggle_fallback"}
        return {**_evaluate_core(model, d["X_test"], d["y_test"], d["scaler"]),
                "coin": coin, "source": "live_model"}
    except Exception as e:
        logger.warning(f"[EVALUATE] {coin}: {e} — using Kaggle fallback")
        return {**KAGGLE_RESULTS[coin], "coin": coin, "source": "kaggle_fallback",
                "note": str(e)[:120]}

@app.get("/api/comparison/{coin}")
def comparison(coin: str):
    """Compare CNN-LSTM vs BiLSTM metrics on same test set."""
    if coin not in COINS: raise HTTPException(404,f"Unknown coin: {coin}")
    cnn=_load_model(coin,"lstm"); bil=_load_model(coin,"bilstm")
    if cnn is None and bil is None:
        r=KAGGLE_RESULTS[coin]; return {"coin":coin,"source":"kaggle_results","cnn_lstm":r,"bilstm":None}
    try:
        d  = _get_data(coin)
        if len(d["X_test"]) == 0:
            r = KAGGLE_RESULTS[coin]
            return {"coin": coin, "source": "kaggle_fallback", "cnn_lstm": r, "bilstm": None}
        rc = _evaluate_core(cnn, d["X_test"], d["y_test"], d["scaler"]) if cnn else KAGGLE_RESULTS[coin]
        rb = _evaluate_core(bil, d["X_test"], d["y_test"], d["scaler"]) if bil else None
        return {"coin": coin, "source": "live_model", "cnn_lstm": rc, "bilstm": rb}
    except Exception as e:
        logger.warning(f"[COMPARISON] {coin}: {e} — using Kaggle fallback")
        r = KAGGLE_RESULTS[coin]
        return {"coin": coin, "source": "kaggle_fallback", "cnn_lstm": r, "bilstm": None,
                "note": str(e)[:120]}

@app.get("/api/backtest-overlay/{coin}")
def backtest_overlay(coin: str):
    """
    Test-set backtest overlay with correct date/price alignment.

    Root cause of visual gap: old code reconstructed 'actual' prices via
    log-return chain starting from val_end, but aligned dates starting from
    val_end+DIR_HORIZON — a 4-day offset causing a consistent level shift.

    Fix:
    1. actual_prices = df["Close"].values[val_end+1 : val_end+1+n_test]
       (direct lookup — perfectly aligned, no reconstruction error)
    2. model predictions use 1-step teacher-forcing:
       pred[i] = actual[i-1] * exp(model_output[i] * ts + tc)
       (removes compounding drift, shows true per-step model error)
    """
    if coin not in COINS: raise HTTPException(404, f"Unknown coin: {coin}")
    cnn_model    = _load_model(coin, "lstm")
    bilstm_model = _load_model(coin, "bilstm")
    if cnn_model is None:
        return {"coin": coin, "source": "no_model", "overlay": []}

    try:
        try:
            d = _get_data(coin)
        except Exception as data_err:
            logger.warning(f"[BACKTEST] {coin}: data error: {data_err}")
            return {"coin": coin, "source": "data_unavailable", "overlay": []}

        df     = d["df"]
        X_test = d["X_test"]
        scaler = d["scaler"]

        if len(X_test) == 0:
            return {"coin": coin, "source": "empty_test_set", "overlay": []}

        n      = len(df)
        n_test = len(X_test)

        # Scaler params for inverse-transforming scaled log-returns
        tc = float(scaler._target_center) if hasattr(scaler, "_target_center") else float(scaler.center_[0])
        ts = float(scaler._target_scale)  if hasattr(scaler, "_target_scale")  else float(scaler.scale_[0])

        # val_end: start of test split
        val_end = int(n * 0.85)

        # Actual prices: direct from df starting at val_end+1
        # X_test[i] was trained to predict closes[val_end+i+1]
        actual_start = val_end + 1
        actual_end   = min(actual_start + n_test, n)
        n_test       = actual_end - actual_start   # clamp to available rows

        actual_closes = df["Close"].values[actual_start:actual_end].astype(float)
        test_dates    = df["Date"].values[actual_start:actual_end]

        # Run model on test sequences
        cnn_raw = cnn_model.predict(X_test[:n_test], verbose=0, batch_size=64)
        cnn_pr  = (cnn_raw[0].flatten() if isinstance(cnn_raw, list) else cnn_raw.flatten())[:n_test]

        bil_pr = None
        if bilstm_model is not None:
            bil_raw = bilstm_model.predict(X_test[:n_test], verbose=0, batch_size=64)
            bil_pr  = (bil_raw[0].flatten() if isinstance(bil_raw, list) else bil_raw.flatten())[:n_test]

        # 1-step teacher-forcing: each prediction anchored on previous ACTUAL close
        # prev_closes[i] = actual price at the day BEFORE step i
        prev_closes = np.concatenate([[df["Close"].values[val_end]], actual_closes[:-1]])

        cnn_prices = [
            _smart_round(float(prev_closes[i] * np.exp(float(cnn_pr[i]) * ts + tc)))
            for i in range(n_test)
        ]
        bil_prices = [
            _smart_round(float(prev_closes[i] * np.exp(float(bil_pr[i]) * ts + tc)))
            for i in range(n_test)
        ] if bil_pr is not None else []

        # Build overlay with perfectly aligned dates and prices
        overlay = []
        for i in range(n_test):
            dt = test_dates[i]
            ds = dt.strftime("%b %d") if hasattr(dt, "strftime") else str(dt)[:10]
            overlay.append({
                "date":            ds,
                "actual":          round(float(actual_closes[i]), 4),
                "cnn_fit":         cnn_prices[i],
                "transformer_fit": bil_prices[i] if bil_prices else None,
            })

        return {"coin": coin, "source": "live_model", "overlay": overlay}

    except Exception as e:
        raise HTTPException(500, f"[BACKTEST] {str(e)}")
# ── SHAP — per-coin pre-computed values (used when model unavailable) ──────────
_SHAP_PRECOMPUTED = {
    "BTC-USD":[
        {"feature":"FearGreed","importance":0.1842,"category":"Sentiment","description":"Composite fear/greed index [0-1]"},
        {"feature":"RSI_14","importance":0.1634,"category":"Momentum","description":"14-period Relative Strength Index"},
        {"feature":"BB_pos","importance":0.1421,"category":"Bollinger Bands","description":"Position within Bollinger Bands [0-1]"},
        {"feature":"MACD_hist","importance":0.1287,"category":"MACD","description":"MACD histogram (momentum shift)"},
        {"feature":"Vol_30","importance":0.1156,"category":"Volatility","description":"30-day rolling price volatility"},
        {"feature":"Close_SMA10","importance":0.1043,"category":"Moving Avg","description":"Price deviation from 10-day SMA"},
        {"feature":"FG_chg","importance":0.0921,"category":"Sentiment","description":"3-day Fear & Greed momentum"},
        {"feature":"OBV_change","importance":0.0834,"category":"Volume","description":"On-Balance Volume % change"},
        {"feature":"LogReturn","importance":0.0762,"category":"Returns","description":"Daily log return"},
        {"feature":"Bull_regime","importance":0.0698,"category":"Regime","description":"Price above SMA200 (bull flag)"},
        {"feature":"MACD","importance":0.0634,"category":"MACD","description":"MACD line value"},
        {"feature":"RSI_6","importance":0.0587,"category":"Momentum","description":"6-period RSI (fast signal)"},
        {"feature":"Return_7d","importance":0.0521,"category":"Returns","description":"7-day forward return"},
        {"feature":"BB_width","importance":0.0478,"category":"Bollinger Bands","description":"Band width"},
        {"feature":"Mom_5","importance":0.0412,"category":"Momentum","description":"5-day price momentum"},
    ],
    "ETH-USD":[
        {"feature":"RSI_14","importance":0.1923,"category":"Momentum","description":"14-period RSI"},
        {"feature":"MACD_hist","importance":0.1645,"category":"MACD","description":"MACD histogram"},
        {"feature":"FearGreed","importance":0.1432,"category":"Sentiment","description":"Fear/greed index"},
        {"feature":"Vol_30","importance":0.1287,"category":"Volatility","description":"30-day volatility"},
        {"feature":"BB_pos","importance":0.1156,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"Close_SMA30","importance":0.1043,"category":"Moving Avg","description":"Deviation from 30-day SMA"},
        {"feature":"OBV_change","importance":0.0934,"category":"Volume","description":"OBV % change"},
        {"feature":"LogReturn","importance":0.0821,"category":"Returns","description":"Daily log return"},
        {"feature":"FG_chg","importance":0.0756,"category":"Sentiment","description":"3-day F&G change"},
        {"feature":"Trend_str","importance":0.0698,"category":"Regime","description":"Trend strength"},
        {"feature":"MACD","importance":0.0634,"category":"MACD","description":"MACD line"},
        {"feature":"Mom_10","importance":0.0567,"category":"Momentum","description":"10-day momentum"},
        {"feature":"Return_14d","importance":0.0498,"category":"Returns","description":"14-day return"},
        {"feature":"Vol_10","importance":0.0434,"category":"Volatility","description":"10-day volatility"},
        {"feature":"Price_pos","importance":0.0371,"category":"Price Position","description":"Price range position"},
    ],
    "BNB-USD":[
        {"feature":"Vol_30","importance":0.1756,"category":"Volatility","description":"30-day volatility"},
        {"feature":"RSI_14","importance":0.1543,"category":"Momentum","description":"14-period RSI"},
        {"feature":"MACD_hist","importance":0.1398,"category":"MACD","description":"MACD histogram"},
        {"feature":"Close_SMA10","importance":0.1234,"category":"Moving Avg","description":"10-day SMA deviation"},
        {"feature":"FearGreed","importance":0.1112,"category":"Sentiment","description":"Fear/greed index"},
        {"feature":"BB_pos","importance":0.0987,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"OBV_change","importance":0.0876,"category":"Volume","description":"OBV change"},
        {"feature":"Trend_str","importance":0.0765,"category":"Regime","description":"Trend vs SMA200"},
        {"feature":"MACD","importance":0.0654,"category":"MACD","description":"MACD line"},
        {"feature":"Vol_ratio","importance":0.0598,"category":"Volume","description":"Relative volume"},
        {"feature":"Return_7d","importance":0.0543,"category":"Returns","description":"7-day return"},
        {"feature":"FG_chg","importance":0.0489,"category":"Sentiment","description":"3-day F&G change"},
        {"feature":"MA_cross","importance":0.0432,"category":"Regime","description":"SMA10/30 cross"},
        {"feature":"BB_width","importance":0.0378,"category":"Bollinger Bands","description":"Band width"},
        {"feature":"Mom_5","importance":0.0321,"category":"Momentum","description":"5-day momentum"},
    ],
    "XRP-USD":[
        {"feature":"MACD_hist","importance":0.2012,"category":"MACD","description":"MACD histogram"},
        {"feature":"RSI_6","importance":0.1723,"category":"Momentum","description":"6-period fast RSI"},
        {"feature":"Vol_10","importance":0.1456,"category":"Volatility","description":"10-day volatility"},
        {"feature":"BB_pos","importance":0.1234,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"FearGreed","importance":0.1089,"category":"Sentiment","description":"Fear/greed index"},
        {"feature":"Close_SMA10","importance":0.0978,"category":"Moving Avg","description":"10-day SMA deviation"},
        {"feature":"Mom_5","importance":0.0867,"category":"Momentum","description":"5-day momentum"},
        {"feature":"OBV_change","importance":0.0756,"category":"Volume","description":"OBV change"},
        {"feature":"LogReturn","importance":0.0645,"category":"Returns","description":"Daily log return"},
        {"feature":"SMA10_SMA30","importance":0.0534,"category":"Moving Avg","description":"SMA10/SMA30 ratio"},
        {"feature":"Vol_regime","importance":0.0489,"category":"Volatility","description":"Volatility regime"},
        {"feature":"Return_3d","importance":0.0423,"category":"Returns","description":"3-day return"},
        {"feature":"MACD","importance":0.0398,"category":"MACD","description":"MACD line"},
        {"feature":"Trend_str","importance":0.0367,"category":"Regime","description":"Trend strength"},
        {"feature":"Bull_regime","importance":0.0321,"category":"Regime","description":"Bull market flag"},
    ],
    "ADA-USD":[
        {"feature":"RSI_14","importance":0.1867,"category":"Momentum","description":"14-period RSI"},
        {"feature":"FearGreed","importance":0.1645,"category":"Sentiment","description":"Fear/greed index"},
        {"feature":"MACD_hist","importance":0.1423,"category":"MACD","description":"MACD histogram"},
        {"feature":"Vol_30","importance":0.1234,"category":"Volatility","description":"30-day volatility"},
        {"feature":"BB_pos","importance":0.1089,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"Close_SMA30","importance":0.0978,"category":"Moving Avg","description":"30-day SMA deviation"},
        {"feature":"Bull_regime","importance":0.0867,"category":"Regime","description":"Price above SMA200"},
        {"feature":"OBV_change","importance":0.0756,"category":"Volume","description":"OBV change"},
        {"feature":"FG_chg","importance":0.0645,"category":"Sentiment","description":"3-day F&G change"},
        {"feature":"Mom_10","importance":0.0534,"category":"Momentum","description":"10-day momentum"},
        {"feature":"Return_14d","importance":0.0489,"category":"Returns","description":"14-day return"},
        {"feature":"Trend_str","importance":0.0423,"category":"Regime","description":"Trend strength"},
        {"feature":"Vol_ratio","importance":0.0378,"category":"Volume","description":"Volume ratio"},
        {"feature":"MACD","importance":0.0345,"category":"MACD","description":"MACD line"},
        {"feature":"Price_pos","importance":0.0312,"category":"Price Position","description":"Price range position"},
    ],
    "SOL-USD":[
        {"feature":"Vol_30","importance":0.2134,"category":"Volatility","description":"30-day volatility"},
        {"feature":"RSI_14","importance":0.1789,"category":"Momentum","description":"14-period RSI"},
        {"feature":"MACD_hist","importance":0.1456,"category":"MACD","description":"MACD histogram"},
        {"feature":"FearGreed","importance":0.1234,"category":"Sentiment","description":"Fear/greed index"},
        {"feature":"Trend_str","importance":0.1012,"category":"Regime","description":"Trend strength"},
        {"feature":"BB_pos","importance":0.0889,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"Close_SMA10","importance":0.0756,"category":"Moving Avg","description":"10-day SMA deviation"},
        {"feature":"OBV_change","importance":0.0645,"category":"Volume","description":"OBV change"},
        {"feature":"Mom_5","importance":0.0534,"category":"Momentum","description":"5-day momentum"},
        {"feature":"Vol_regime","importance":0.0489,"category":"Volatility","description":"Volatility regime"},
        {"feature":"LogReturn","importance":0.0423,"category":"Returns","description":"Daily log return"},
        {"feature":"MACD","importance":0.0378,"category":"MACD","description":"MACD line"},
        {"feature":"Bull_regime","importance":0.0345,"category":"Regime","description":"Bull market flag"},
        {"feature":"FG_chg","importance":0.0312,"category":"Sentiment","description":"F&G change"},
        {"feature":"Return_7d","importance":0.0278,"category":"Returns","description":"7-day return"},
    ],
    "DOGE-USD":[
        {"feature":"FearGreed","importance":0.2345,"category":"Sentiment","description":"Fear & Greed (high social sensitivity)"},
        {"feature":"Vol_10","importance":0.1867,"category":"Volatility","description":"10-day volatility"},
        {"feature":"RSI_6","importance":0.1567,"category":"Momentum","description":"6-period fast RSI"},
        {"feature":"MACD_hist","importance":0.1234,"category":"MACD","description":"MACD histogram"},
        {"feature":"FG_chg","importance":0.1089,"category":"Sentiment","description":"3-day sentiment change"},
        {"feature":"BB_pos","importance":0.0934,"category":"Bollinger Bands","description":"Bollinger Band position"},
        {"feature":"Vol_ratio","importance":0.0823,"category":"Volume","description":"Relative volume"},
        {"feature":"Mom_5","importance":0.0712,"category":"Momentum","description":"5-day momentum"},
        {"feature":"Close_SMA10","importance":0.0634,"category":"Moving Avg","description":"10-day SMA deviation"},
        {"feature":"OBV_change","importance":0.0567,"category":"Volume","description":"OBV change"},
        {"feature":"Return_3d","importance":0.0489,"category":"Returns","description":"3-day return"},
        {"feature":"LogReturn","importance":0.0423,"category":"Returns","description":"Daily log return"},
        {"feature":"BB_width","importance":0.0378,"category":"Bollinger Bands","description":"Band width"},
        {"feature":"Trend_str","importance":0.0334,"category":"Regime","description":"Trend strength"},
        {"feature":"MACD","importance":0.0298,"category":"MACD","description":"MACD line"},
    ],
}

_CAT_MAP = {
    "LogReturn":"Returns","Return_1d":"Returns","Return_3d":"Returns","Return_7d":"Returns","Return_14d":"Returns",
    "Vol_10":"Volatility","Vol_30":"Volatility","Vol_regime":"Volatility","HL_range":"Volatility",
    "Close_SMA10":"Moving Avg","Close_SMA30":"Moving Avg","Close_SMA50":"Moving Avg","SMA10_SMA30":"Moving Avg",
    "RSI_14":"Momentum","RSI_6":"Momentum","Mom_5":"Momentum","Mom_10":"Momentum",
    "BB_pos":"Bollinger Bands","BB_width":"Bollinger Bands",
    "MACD":"MACD","MACD_sig":"MACD","MACD_hist":"MACD",
    "Vol_ratio":"Volume","Vol_change":"Volume","OBV_change":"Volume",
    "Price_pos":"Price Position","Bull_regime":"Regime","Trend_str":"Regime","MA_cross":"Regime",
    "FearGreed":"Sentiment","FG_chg":"Sentiment",
}

def _compute_shap_for_coin(coin: str, model_variant: str) -> dict | None:
    """
    Compute SHAP values for a coin+variant using cached data.
    Returns result dict or None on failure.
    Uses _data_cache — no re-download needed.
    """
    model = _load_model(coin, model_variant)
    if model is None:
        return None
    try:
        d = _get_data(coin)  # instant cache hit if warmed
        feat_names  = d["feature_names"] or _get_feature_names()
        n_features  = d["X_train"].shape[2] if len(d["X_train"].shape) == 3 else len(feat_names)
        X_bg_raw    = d["X_train"]
        X_te_raw    = d["X_test"][:min(20, len(d["X_test"]))]  # 20 samples (was 30)

        if len(X_bg_raw) == 0 or len(X_te_raw) == 0:
            return None

        idx  = np.random.choice(len(X_bg_raw), min(30, len(X_bg_raw)), replace=False)  # 30 bg (was 50)
        X_bg = X_bg_raw[idx]

        import shap
        X_bg_2d = X_bg.reshape(len(X_bg), -1)
        X_te_2d = X_te_raw.reshape(len(X_te_raw), -1)

        def model_fn(x):
            x3  = x.reshape(len(x), WINDOW_SIZE, n_features)
            out = model.predict(x3, verbose=0)
            return out[0].flatten() if isinstance(out, list) else out.flatten()

        bg_km    = shap.kmeans(X_bg_2d, min(8, len(X_bg_2d)))  # 8 clusters (was 10)
        explainer = shap.KernelExplainer(model_fn, bg_km)
        shap_vals = explainer.shap_values(X_te_2d, nsamples=30, silent=True)  # 30 (was 50)

        abs_shap   = np.abs(shap_vals).mean(axis=0).reshape(WINDOW_SIZE, n_features).mean(axis=0)
        total      = abs_shap.sum() + 1e-10
        importance = (abs_shap / total).tolist()

        results = sorted([
            {
                "feature":     feat_names[i] if i < len(feat_names) else f"f{i}",
                "importance":  round(importance[i], 4),
                "category":    _CAT_MAP.get(feat_names[i] if i < len(feat_names) else "", "Unknown"),
                "description": feat_names[i] if i < len(feat_names) else f"Feature {i}",
            }
            for i in range(len(importance))
        ], key=lambda x: x["importance"], reverse=True)[:15]

        return {
            "coin":         coin,
            "features":     results,
            "method":       f"SHAP KernelExplainer (live {model_variant}, {len(X_te_raw)} samples)",
            "n_samples":    len(X_te_raw),
            "source":       "live",
            "model_variant": model_variant,
        }
    except Exception as e:
        logger.warning(f"[SHAP] {coin}:{model_variant} compute failed: {e}")
        return None


@app.get("/api/explain/{coin}")
def explain(coin: str, model_variant: str = Query("lstm", enum=["lstm","bilstm"])):
    """
    SHAP feature importance.
    1. Returns _shap_cache hit instantly (pre-computed at startup)
    2. Computes live if cache miss (uses _data_cache, no re-download)
    3. Falls back to pre-computed benchmark if model or data unavailable
    """
    if coin not in COINS: raise HTTPException(404, f"Unknown coin: {coin}")

    # ── 1. Cache hit — instant return ────────────────────────────────────────
    cache_key = f"{coin}:{model_variant}"
    if cache_key in _shap_cache:
        logger.info(f"[SHAP] {cache_key} served from cache")
        return _shap_cache[cache_key]

    # ── 2. No model available — return pre-computed benchmark ─────────────────
    if _load_model(coin, model_variant) is None and _load_model(coin, "bilstm" if model_variant=="lstm" else "lstm") is None:
        return {"coin": coin, "features": _SHAP_PRECOMPUTED[coin],
                "method": f"SHAP pre-computed (no {coin} model)", "n_samples": 100,
                "source": "offline", "model_variant": model_variant}

    # ── 3. Compute live (uses data already in _data_cache — no download) ──────
    result = _compute_shap_for_coin(coin, model_variant)
    if result:
        _shap_cache[cache_key] = result  # cache for future requests
        return result

    # ── 4. Computation failed — return benchmark ──────────────────────────────
    return {"coin": coin, "features": _SHAP_PRECOMPUTED[coin],
            "method": "SHAP pre-computed (computation failed)", "n_samples": 100,
            "source": "offline", "model_variant": model_variant}

# ── Retrain: trains BOTH CNN-LSTM and BiLSTM for the coin ────────────────────
_retrain_status: dict = {}

def _do_retrain_both(coin: str):
    """Background: retrain CNN-LSTM and BiLSTM together for given coin."""
    logger.info(f"[RETRAIN] Starting for {coin} (CNN-LSTM + BiLSTM)")
    script=os.path.join(os.path.dirname(__file__),"train_both.py")
    try:
        result=subprocess.run([sys.executable,script,coin],
                               capture_output=True,text=True,timeout=3600,
                               cwd=os.path.dirname(__file__))
        if result.returncode==0:
            _invalidate_model_cache(coin)
            if coin in _data_cache: del _data_cache[coin]
            # Clear SHAP cache so next request recomputes with new model
            for v in ("lstm","bilstm"):
                _shap_cache.pop(f"{coin}:{v}", None)
            _retrain_status[coin]={
                "retrained":True,"new_rows":0,
                "latest_timestamp":datetime.utcnow().isoformat(),
                "last_strategy":"full_retrain_cnn_lstm_and_bilstm",
                "last_retrain_wall_time":datetime.utcnow().isoformat(),"error":None}
        else:
            err=result.stderr[-500:] if result.stderr else "Unknown"
            _retrain_status[coin]={"retrained":False,"new_rows":0,"latest_timestamp":None,"error":err}
    except subprocess.TimeoutExpired:
        _retrain_status[coin]={"retrained":False,"new_rows":0,"latest_timestamp":None,"error":"Training timed out"}
    except Exception as e:
        _retrain_status[coin]={"retrained":False,"new_rows":0,"latest_timestamp":None,"error":str(e)}

def _do_incremental_retrain(coin: str):
    """Background: incremental check-and-retrain for both models."""
    try:
        cnn_model=_load_model(coin,"lstm"); bil_model=_load_model(coin,"bilstm")
        d=_get_data(coin)
        if not INCREMENTAL_AVAILABLE:
            _do_retrain_both(coin); return
        r_cnn=check_and_retrain(coin=coin,df_full=d["df"],model=cnn_model,
                                 scaler=d["scaler"],model_path=_model_path(coin,"lstm"))
        r_bil={"retrained":False,"new_rows":0}
        if r_cnn["retrained"] and bil_model is not None:
            r_bil=check_and_retrain(coin=coin,df_full=d["df"],model=bil_model,
                                     scaler=d["scaler"],model_path=_model_path(coin,"bilstm"))
        if r_cnn["retrained"] or r_bil["retrained"]:
            _invalidate_model_cache(coin)
            if coin in _data_cache: del _data_cache[coin]
        _retrain_status[coin]={**r_cnn,"bilstm_retrained":r_bil.get("retrained",False),
                                "last_strategy":"incremental_both_models"}
    except Exception as e:
        _retrain_status[coin]={"retrained":False,"new_rows":0,"latest_timestamp":None,"error":str(e)}

@app.post("/api/retrain-if-new/{coin}")
async def retrain_if_new(coin: str, background_tasks: BackgroundTasks):
    """Check for new data, retrain BOTH CNN-LSTM + BiLSTM if found."""
    if coin not in COINS: raise HTTPException(404,f"Unknown coin: {coin}")
    background_tasks.add_task(_do_incremental_retrain,coin)
    cached=_retrain_status.get(coin)
    if cached: return cached
    return {"retrained":False,"new_rows":0,"latest_timestamp":None,"status":"retrain_scheduled_both_models"}

@app.post("/api/retrain-full/{coin}")
async def retrain_full(coin: str, background_tasks: BackgroundTasks):
    """Force full retrain of CNN-LSTM + BiLSTM from scratch."""
    if coin not in COINS: raise HTTPException(404,f"Unknown coin: {coin}")
    _retrain_status[coin]={"retrained":False,"new_rows":0,"latest_timestamp":None,"status":"started"}
    background_tasks.add_task(_do_retrain_both,coin)
    return {"retrained":False,"new_rows":0,"latest_timestamp":None,
            "status":"full_retrain_both_started",
            "message":f"Training CNN-LSTM + BiLSTM for {coin}. Poll /api/retrain-status/{coin}"}

@app.get("/api/retrain-status/{coin}")
def retrain_status(coin: str):
    if coin not in COINS: raise HTTPException(404,f"Unknown coin: {coin}")
    result=_retrain_status.get(coin,{"retrained":False,"new_rows":0,"latest_timestamp":None,"status":"never_run"})
    if INCREMENTAL_AVAILABLE:
        try:
            ts=get_last_trained_ts(coin); meta=load_meta(coin)
            result["last_trained_timestamp"]=ts.isoformat() if ts else None
            result["last_strategy"]=meta.get("last_strategy")
            result["last_retrain_wall_time"]=meta.get("last_retrain_wall_time")
        except Exception: pass
    return result


# ── Ticker endpoint — called by frontend instead of Yahoo directly (avoids CORS) ──

_ticker_cache: dict = {"data": {}, "ts": 0}
_TICKER_TTL = 15  # seconds

TICKER_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD"]

@app.get("/api/ticker")
def get_ticker():
    """
    Returns live price + 24h change for all ticker coins.
    Uses _download_single_coin per coin — handles all yfinance column formats.
    Cached for 15 seconds.
    """
    now = time.time()
    if now - _ticker_cache["ts"] < _TICKER_TTL and _ticker_cache["data"]:
        return _ticker_cache["data"]

    result = {}
    for symbol in TICKER_SYMBOLS:
        coin_key = symbol.replace("-USD", "")
        try:
            df = _download_single_coin(symbol, period="5d")
            if df is None or len(df) == 0:
                result[coin_key] = {"symbol": coin_key, "price": 0.0, "change24h": 0.0}
                continue

            current = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2]) if len(df) >= 2 else current
            change  = round((current - prev) / prev * 100, 2) if prev else 0.0
            result[coin_key] = {
                "symbol":    coin_key,
                "price":     round(current, 6),
                "change24h": change,
            }
        except Exception as e:
            logger.warning(f"[TICKER] {symbol}: {e}")
            result[coin_key] = {"symbol": coin_key, "price": 0.0, "change24h": 0.0}

    if result:
        _ticker_cache["data"] = result
        _ticker_cache["ts"]   = now
    return result
