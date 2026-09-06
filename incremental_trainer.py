"""
incremental_trainer.py
─────────────────────
Adaptive incremental retraining for NeuralPredict.

Strategy
--------
• Detect latest saved training timestamp per coin (stored in models/<coin>_meta.json).
• Compare with the latest market candle timestamp from yfinance.
• If new data exists (even 1 row), trigger retraining:
  - 1–3 new rows → fast fine-tuning only (small epochs, frozen base layers).
  - 4+ new rows  → full incremental retrain (all layers, more epochs).
• Appends new rows to existing dataset before fitting.
• Reuses/preserves existing model weights (fine-tune from checkpoint).
• Saves updated model, scaler metadata, and last-trained timestamp.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("incremental_trainer")

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
WINDOW_SIZE = 30
DIR_HORIZON = 5
STRONG_QUANTILE = 0.85

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Metadata helpers ────────────────────────────────────────────────────────

def _meta_path(coin: str) -> str:
    return os.path.join(MODEL_DIR, f"{coin}_meta.json")


def load_meta(coin: str) -> dict:
    p = _meta_path(coin)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_meta(coin: str, meta: dict):
    with open(_meta_path(coin), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def get_last_trained_ts(coin: str) -> Optional[pd.Timestamp]:
    """Return the last trained timestamp for a coin, or None if never trained."""
    meta = load_meta(coin)
    ts_str = meta.get("last_trained_timestamp")
    if ts_str:
        try:
            return pd.Timestamp(ts_str)
        except Exception:
            pass
    return None


def set_last_trained_ts(coin: str, ts: pd.Timestamp):
    meta = load_meta(coin)
    meta["last_trained_timestamp"] = ts.isoformat()
    meta["last_retrain_wall_time"] = datetime.utcnow().isoformat()
    save_meta(coin, meta)


# ─── Feature engineering (mirrors main.py) ──────────────────────────────────

def _get_feature_names():
    return [
        "LogReturn","Return_1d","Return_3d","Return_7d","Return_14d",
        "Vol_10","Vol_30","HL_range","Close_SMA10","Close_SMA30","Close_SMA50","SMA10_SMA30",
        "RSI_14","RSI_6","BB_pos","BB_width","MACD","MACD_sig","MACD_hist",
        "Mom_5","Mom_10","Vol_ratio","Vol_change","OBV_change","Price_pos",
        "Bull_regime","Trend_str","MA_cross","Vol_regime","FearGreed","FG_chg",
    ]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["LogReturn"]  = np.log(data["Close"] / data["Close"].shift(1))
    data["Return_1d"]  = data["Close"].pct_change(1)
    data["Return_3d"]  = data["Close"].pct_change(3)
    data["Return_7d"]  = data["Close"].pct_change(7)
    data["Return_14d"] = data["Close"].pct_change(14)
    data["Vol_10"]     = data["LogReturn"].rolling(10).std()
    data["Vol_30"]     = data["LogReturn"].rolling(30).std()
    data["HL_range"]   = (data["High"] - data["Low"]) / (data["Close"] + 1e-10)

    sma10  = data["Close"].rolling(10).mean()
    sma30  = data["Close"].rolling(30).mean()
    sma50  = data["Close"].rolling(50).mean()
    sma200 = data["Close"].rolling(200).mean()
    ema12  = data["Close"].ewm(span=12, adjust=False).mean()
    ema26  = data["Close"].ewm(span=26, adjust=False).mean()

    data["Close_SMA10"] = data["Close"] / (sma10  + 1e-10) - 1
    data["Close_SMA30"] = data["Close"] / (sma30  + 1e-10) - 1
    data["Close_SMA50"] = data["Close"] / (sma50  + 1e-10) - 1
    data["SMA10_SMA30"] = sma10 / (sma30 + 1e-10) - 1
    data["Bull_regime"] = (data["Close"] > sma200).astype(float)
    data["Trend_str"]   = data["Close"] / (sma200 + 1e-10) - 1
    data["MA_cross"]    = (sma10 > sma30).astype(float)
    data["Vol_regime"]  = (
        data["LogReturn"].rolling(30).std() /
        (data["LogReturn"].rolling(90).std() + 1e-10) - 1
    )

    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    data["RSI_14"] = (100 - (100 / (1 + gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)))) / 50 - 1
    data["RSI_6"]  = (100 - (100 / (1 + gain.rolling(6).mean()  / (loss.rolling(6).mean()  + 1e-10)))) / 50 - 1

    bb_mid = data["Close"].rolling(20).mean()
    bb_std = data["Close"].rolling(20).std()
    data["BB_pos"]   = (data["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    data["BB_width"] = (4*bb_std) / (bb_mid + 1e-10)

    data["MACD"]      = (ema12 - ema26) / (data["Close"] + 1e-10)
    data["MACD_sig"]  = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_sig"]
    data["Mom_5"]     = data["Close"].pct_change(5)
    data["Mom_10"]    = data["Close"].pct_change(10)

    vm = data["Volume"].rolling(10).mean()
    data["Vol_ratio"]  = data["Volume"] / (vm + 1e-10) - 1
    data["Vol_change"] = data["Volume"].pct_change()
    data["Price_pos"]  = (
        (data["Close"] - data["Close"].rolling(20).min()) /
        (data["Close"].rolling(20).max() - data["Close"].rolling(20).min() + 1e-10)
    )

    obv = [0]
    for i in range(1, len(data)):
        if   data["Close"].iloc[i] > data["Close"].iloc[i-1]: obv.append(obv[-1] + data["Volume"].iloc[i])
        elif data["Close"].iloc[i] < data["Close"].iloc[i-1]: obv.append(obv[-1] - data["Volume"].iloc[i])
        else: obv.append(obv[-1])
    data["OBV_change"] = pd.Series(obv).pct_change().fillna(0).values

    if "FearGreed" not in data.columns: data["FearGreed"] = 0.5
    if "FG_chg"    not in data.columns: data["FG_chg"]    = 0.0

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def _build_sequences(scaled_arr, closes, fwd_values, window=WINDOW_SIZE, dh=DIR_HORIZON):
    """Build (X, y_return, y_direction) sequence arrays."""
    up_thr = np.quantile(fwd_values[~np.isnan(fwd_values)], STRONG_QUANTILE)
    dn_thr = np.quantile(fwd_values[~np.isnan(fwd_values)], 1 - STRONG_QUANTILE)

    STRONG_UP, NEUTRAL, STRONG_DOWN = 2, 1, 0

    def label(r):
        if r > up_thr:  return STRONG_UP
        if r < dn_thr:  return STRONG_DOWN
        return NEUTRAL

    X, yr, y3 = [], [], []
    for i in range(window, len(scaled_arr) - dh):
        f = fwd_values[i]
        if np.isnan(f): continue
        X.append(scaled_arr[i - window:i])
        yr.append(scaled_arr[i, 0])
        y3.append(label(f))
    return np.array(X), np.array(yr), np.array(y3)


# ─── Core retraining function ────────────────────────────────────────────────

def check_and_retrain(
    coin: str,
    df_full: pd.DataFrame,
    model,
    scaler,
    model_path: str,
) -> dict:
    """
    Check if there's new data since last training and retrain if so.

    Parameters
    ----------
    coin       : coin id (e.g. "BTC-USD")
    df_full    : full preprocessed DataFrame from yfinance (with FearGreed cols)
    model      : loaded Keras model (or None if not yet saved)
    scaler     : fitted RobustScaler from last training
    model_path : path to save updated .keras model

    Returns
    -------
    dict with keys: retrained, new_rows, latest_timestamp
    """
    import tensorflow as tf

    # Latest market candle date
    latest_market_ts = pd.Timestamp(df_full["Date"].max())
    last_trained_ts  = get_last_trained_ts(coin)

    result = {
        "retrained":        False,
        "new_rows":         0,
        "latest_timestamp": latest_market_ts.isoformat(),
    }

    # Determine how many new candles exist
    if last_trained_ts is not None:
        new_df = df_full[df_full["Date"] > last_trained_ts]
        new_rows = len(new_df)
    else:
        # Never trained — treat all as new (triggers full retrain)
        new_rows = len(df_full)

    result["new_rows"] = new_rows

    if new_rows == 0:
        logger.info(f"[{coin}] No new data. Skipping retrain.")
        return result

    logger.info(f"[{coin}] {new_rows} new row(s) since last train — triggering retrain.")

    # ── Feature engineering on full dataset ─────────────────────────────────
    FCOLS = _get_feature_names()
    data  = _engineer_features(df_full)

    if len(data) < WINDOW_SIZE + DIR_HORIZON + 10:
        logger.warning(f"[{coin}] Not enough rows after feature engineering ({len(data)}). Skipping.")
        return result

    # ── Rescale with existing scaler (no refit — preserves distribution) ────
    from sklearn.preprocessing import RobustScaler

    feat_arr = data[FCOLS].values
    closes   = data["Close"].values
    n        = len(closes)

    # Refit scaler only on 70% train slice (consistent with original training)
    train_end = int(n * 0.70)
    try:
        scaler.fit(feat_arr[:train_end])   # update scaler on new full history
    except Exception as e:
        logger.warning(f"[{coin}] Scaler refit failed: {e}. Using existing scaler.")

    scaled = scaler.transform(feat_arr)

    fwd = np.array([
        (closes[i + DIR_HORIZON] - closes[i]) / (closes[i] + 1e-10)
        if i + DIR_HORIZON < n else np.nan
        for i in range(n)
    ])

    # ── Build sequences ──────────────────────────────────────────────────────
    X, yr, y3 = _build_sequences(scaled, closes, fwd)

    if len(X) == 0:
        logger.warning(f"[{coin}] 0 sequences built. Skipping retrain.")
        return result

    # ── Decide training strategy ─────────────────────────────────────────────
    if new_rows <= 3:
        # Fast fine-tune: only last 10% of sequences, few epochs, high LR
        n_finetune = max(32, int(len(X) * 0.10))
        X_ft = X[-n_finetune:]
        yr_ft = yr[-n_finetune:]
        epochs    = 3
        batch_sz  = min(16, len(X_ft))
        freeze_base = True
        strategy = "fast_finetune"
    else:
        # Full incremental retrain on all sequences
        X_ft  = X
        yr_ft = yr
        epochs    = max(5, min(15, new_rows * 2))
        batch_sz  = 32
        freeze_base = False
        strategy = "full_retrain"

    logger.info(f"[{coin}] Strategy={strategy}, sequences={len(X_ft)}, epochs={epochs}")

    # ── Compile and fit ───────────────────────────────────────────────────────
    try:
        if model is None:
            logger.warning(f"[{coin}] No model loaded — cannot retrain. Skipping.")
            return result

        # Optionally freeze early layers for fast fine-tune
        if freeze_base and len(model.layers) > 4:
            for layer in model.layers[:-4]:
                layer.trainable = False
        else:
            for layer in model.layers:
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 if freeze_base else 5e-5),
            loss="mse",
        )

        model.fit(
            X_ft, yr_ft,
            epochs=epochs,
            batch_size=batch_sz,
            verbose=0,
            shuffle=True,
        )

        # Re-enable all layers after fine-tune
        for layer in model.layers:
            layer.trainable = True

        # ── Save model ───────────────────────────────────────────────────────
        model.save(model_path)
        logger.info(f"[{coin}] Model saved → {model_path}")

        # ── Persist scaler metadata ──────────────────────────────────────────
        scaler._target_center     = float(scaler.center_[0])
        scaler._target_scale      = float(scaler.scale_[0])
        scaler._up_thr            = float(np.quantile(fwd[~np.isnan(fwd)], STRONG_QUANTILE))
        scaler._down_thr          = float(np.quantile(fwd[~np.isnan(fwd)], 1 - STRONG_QUANTILE))
        val_end = int(n * 0.85)
        scaler._closes_test_start = float(closes[val_end]) if val_end < n else float(closes[-1])

        # ── Save metadata ────────────────────────────────────────────────────
        set_last_trained_ts(coin, latest_market_ts)
        meta = load_meta(coin)
        meta.update({
            "last_strategy":     strategy,
            "last_new_rows":     new_rows,
            "total_sequences":   len(X),
            "latest_market_date": latest_market_ts.isoformat(),
        })
        save_meta(coin, meta)

        result["retrained"] = True
        logger.info(f"[{coin}] ✅ Retrain complete. Strategy={strategy}, new_rows={new_rows}")

    except Exception as e:
        logger.error(f"[{coin}] Retrain failed: {e}")

    return result
