"""
train_both.py — Train BOTH CNN-LSTM and BiLSTM for one (or all) coins.

Usage:
    python train_both.py              # train all 7 coins, both models each
    python train_both.py BTC-USD      # train both models for BTC only
    python train_both.py ETH-USD SOL-USD   # train both models for listed coins

Each coin produces:
    models/{coin}_lstm.keras    — CNN-LSTM model
    models/{coin}_bilstm.keras  — BiLSTM model

Architecture summary
────────────────────
CNN-LSTM  : Conv1D(64,3) → MaxPool → LSTM(96) → Dense(64) → [price, direction(3)]
BiLSTM    : BiLSTM(128) → BiLSTM(64) → Attention → Dense(64) → [price, direction(3)]
Both use dual outputs: price regression (Huber) + direction classification (Focal γ=2)
"""

import os, sys, json, time
import numpy as np
import pandas as pd

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ALL_COINS    = ["BTC-USD","ETH-USD","XRP-USD","ADA-USD","BNB-USD","SOL-USD","DOGE-USD"]
WINDOW_SIZE  = 30
DIR_HORIZON  = 5
EPOCHS_MAIN  = 80    # reduced for retrain context; full training uses more
EPOCHS_FINE  = 20
BATCH_SIZE   = 32
LR_MAIN      = 0.001
LR_FINE      = 0.0001
STRONG_Q     = 0.85

# ── Imports ───────────────────────────────────────────────────────────────────
import yfinance as yf
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D,
    Attention, GlobalAveragePooling1D, Flatten, MultiHeadAttention,
    LayerNormalization, Add, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
import requests


# ── Focal loss ────────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred   = K.clip(y_pred, 1e-7, 1.0-1e-7)
        y_true_i = K.cast(y_true, "int32")
        n_cls    = K.shape(y_pred)[1]
        y_oh     = K.cast(K.one_hot(y_true_i, n_cls), "float32")
        ce       = -K.sum(y_oh * K.log(y_pred), axis=-1)
        p_t      = K.sum(y_oh * y_pred, axis=-1)
        fw       = K.pow(1.0-p_t, gamma)
        return K.mean(fw * ce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ── Build CNN-LSTM model ──────────────────────────────────────────────────────
def build_cnn_lstm(input_shape):
    """
    CNN-LSTM hybrid:
      Conv1D(64,3) → MaxPool → Conv1D(128,3) → LSTM(96,seq) → LSTM(64) →
      Dense(128) → Dense(64) → [price_out, direction_out(3)]
    """
    inputs = Input(shape=input_shape)

    # CNN branch — extracts local patterns
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)

    # LSTM branch — temporal context
    x = LSTM(96, return_sequences=True)(x)
    x = Dropout(0.25)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Shared dense
    shared = Dense(128, activation="relu")(x)
    shared = Dropout(0.2)(shared)
    shared = Dense(64, activation="relu")(shared)

    # Price output (regression)
    price = Dense(32, activation="relu")(shared)
    price = Dense(1, name="price_output")(price)

    # Direction output (3-class: down / neutral / up)
    direction = Dense(32, activation="relu")(shared)
    direction = Dense(3, activation="softmax", name="direction_output")(direction)

    return Model(inputs=inputs, outputs=[price, direction])


# ── Build BiLSTM model ────────────────────────────────────────────────────────
def build_bilstm(input_shape):
    """
    Bidirectional LSTM with self-attention:
      BiLSTM(128,seq) → Dropout → BiLSTM(64,seq) → Dropout →
      Attention → GlobalAvgPool → Dense(128) → Dense(64) → [price_out, direction_out(3)]
    """
    inputs = Input(shape=input_shape)

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Self-attention
    attn_out = Attention()([x, x])
    pooled   = GlobalAveragePooling1D()(attn_out)

    # Shared dense
    shared = Dense(128, activation="relu")(pooled)
    shared = Dropout(0.2)(shared)
    shared = Dense(64, activation="relu")(shared)

    # Price output
    price = Dense(32, activation="relu")(shared)
    price = Dense(1, name="price_output")(price)

    # Direction output (3-class)
    direction = Dense(32, activation="relu")(shared)
    direction = Dense(3, activation="softmax", name="direction_output")(direction)

    return Model(inputs=inputs, outputs=[price, direction])


def compile_model(model, lr=0.001, dir_weight=0.4):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={
            "price_output":     "huber",
            "direction_output": focal_loss(gamma=2.0),
        },
        loss_weights={"price_output": 1.0, "direction_output": dir_weight},
        metrics={"price_output": "mae", "direction_output": "accuracy"},
    )
    return model


# ── Feature engineering (mirrors main.py exactly) ────────────────────────────
def build_features(df):
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
    vm=data["Volume"].rolling(10).mean()
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
    FCOLS = [
        "LogReturn","Return_1d","Return_3d","Return_7d","Return_14d",
        "Vol_10","Vol_30","HL_range","Close_SMA10","Close_SMA30","Close_SMA50","SMA10_SMA30",
        "RSI_14","RSI_6","BB_pos","BB_width","MACD","MACD_sig","MACD_hist",
        "Mom_5","Mom_10","Vol_ratio","Vol_change","OBV_change","Price_pos",
        "Bull_regime","Trend_str","MA_cross","Vol_regime","FearGreed","FG_chg",
    ]
    data.dropna(inplace=True); data.reset_index(drop=True,inplace=True)
    return data, FCOLS


def prepare_data(df):
    data, FCOLS = build_features(df)
    closes = data["Close"].values; n=len(closes)
    train_end=int(n*0.70); val_end=int(n*0.85)
    sc=RobustScaler()
    tr_sc=sc.fit_transform(data[FCOLS].values[:train_end])
    va_sc=sc.transform(data[FCOLS].values[train_end:val_end])
    te_sc=sc.transform(data[FCOLS].values[val_end:])

    fwd=np.array([(closes[i+DIR_HORIZON]-closes[i])/(closes[i]+1e-10)
                  if i+DIR_HORIZON<n else np.nan for i in range(n)])
    tr_clean=fwd[:train_end][~np.isnan(fwd[:train_end])]
    up_thr=np.quantile(tr_clean,STRONG_Q); dn_thr=np.quantile(tr_clean,1-STRONG_Q)
    def l3(r): return 2 if r>up_thr else 0 if r<dn_thr else 1

    def mseq(arr, fv):
        X,yr,y3=[],[],[]
        for i in range(WINDOW_SIZE, len(arr)-DIR_HORIZON):
            f=fv[i]
            if np.isnan(f): continue
            X.append(arr[i-WINDOW_SIZE:i]); yr.append(arr[i,0]); y3.append(l3(f))
        return np.array(X),np.array(yr),np.array(y3)

    X_tr,y_tr_p,y_tr_d = mseq(tr_sc, fwd[:train_end])
    ctst=np.concatenate([va_sc[-WINDOW_SIZE:],te_sc])
    ft=np.concatenate([fwd[val_end-WINDOW_SIZE:val_end],fwd[val_end:]])
    X_ts,y_ts_p,y_ts_d = mseq(ctst, ft)

    print(f"  Data ready: X_train={X_tr.shape}, X_test={X_ts.shape}, features={X_tr.shape[2]}")
    return X_tr,X_ts,y_tr_p,y_ts_p,y_tr_d,y_ts_d,sc


def _fetch_fg(df):
    try:
        r=requests.get("https://api.alternative.me/fng/",params={"limit":0,"format":"json"},timeout=15)
        tmp=pd.DataFrame(r.json()["data"])
        tmp["Date"]=pd.to_datetime(tmp["timestamp"].astype(int),unit="s").dt.normalize().dt.tz_localize(None)
        tmp["FearGreed"]=tmp["value"].astype(float)/100.0
        fg=tmp[["Date","FearGreed"]].sort_values("Date").reset_index(drop=True)
        dn=df["Date"].dt.normalize().dt.tz_localize(None)
        df["FearGreed"]=dn.map(dict(zip(fg["Date"],fg["FearGreed"]))).fillna(0.5)
        df["FG_chg"]=df["FearGreed"].diff(3).fillna(0)
    except Exception:
        df["FearGreed"]=0.5; df["FG_chg"]=0.0
    return df


def _normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col_map={}
    for c in df.columns:
        cl=str(c).lower().strip()
        if cl=="open": col_map[c]="Open"
        elif cl=="high": col_map[c]="High"
        elif cl=="low": col_map[c]="Low"
        elif cl=="close": col_map[c]="Close"
        elif cl=="volume": col_map[c]="Volume"
        elif cl in ("date","datetime","index"): col_map[c]="Date"
    df.rename(columns=col_map,inplace=True)
    keep={"Date","Open","High","Low","Close","Volume"}
    drop=[c for c in df.columns if c not in keep]
    if drop: df.drop(columns=drop,inplace=True)
    return df


def train_coin(coin: str) -> dict:
    COIN_START = {
        "BTC-USD":"2018-01-01","ETH-USD":"2018-01-01","BNB-USD":"2018-01-01",
        "XRP-USD":"2018-01-01","ADA-USD":"2018-01-01","SOL-USD":"2020-03-01","DOGE-USD":"2018-01-01",
    }
    print(f"\n{'='*60}")
    print(f"  Training BOTH models for: {coin}")
    print(f"{'='*60}")
    t0=time.time()

    # ── Fetch data ────────────────────────────────────────────────────────
    print(f"  [1/4] Fetching data from {COIN_START.get(coin,'2018-01-01')}...")
    try:
        df=yf.download(coin,start=COIN_START.get(coin,"2018-01-01"),progress=False,auto_adjust=True)
        df.reset_index(inplace=True); df=_normalize(df)
        required=["Date","Open","High","Low","Close","Volume"]
        df=df[required]; df["Date"]=pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.dropna(inplace=True); df.reset_index(drop=True,inplace=True)
        df=_fetch_fg(df)
        print(f"  Loaded {len(df):,} rows ({df['Date'].min().date()} → {df['Date'].max().date()})")
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}"); return {"coin":coin,"status":"failed","error":str(e)}

    # ── Prepare sequences ─────────────────────────────────────────────────
    print(f"  [2/4] Building features & sequences...")
    try:
        X_tr,X_ts,y_tr_p,y_ts_p,y_tr_d,y_ts_d,sc = prepare_data(df)
    except Exception as e:
        print(f"  ❌ Preprocessing failed: {e}"); return {"coin":coin,"status":"failed","error":str(e)}

    input_shape = (X_tr.shape[1], X_tr.shape[2])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=0),
    ]

    results = {}

    # ── Train CNN-LSTM ────────────────────────────────────────────────────
    print(f"\n  [3/4] Training CNN-LSTM...")
    try:
        cnn = build_cnn_lstm(input_shape)
        cnn = compile_model(cnn, lr=LR_MAIN, dir_weight=0.4)
        cnn.summary(print_fn=lambda x: None)  # silent summary

        t1=time.time()
        hist1=cnn.fit(X_tr,{"price_output":y_tr_p,"direction_output":y_tr_d},
                      epochs=EPOCHS_MAIN,batch_size=BATCH_SIZE,validation_split=0.1,
                      callbacks=callbacks,verbose=0)
        print(f"  CNN-LSTM phase 1 done in {(time.time()-t1)/60:.1f}min")

        # Fine-tune on test data
        cnn=compile_model(cnn,lr=LR_FINE,dir_weight=0.8)
        if len(X_ts)>0:
            cnn.fit(X_ts,{"price_output":y_ts_p,"direction_output":y_ts_d},
                    epochs=EPOCHS_FINE,batch_size=16,validation_split=0.1,verbose=0)

        cnn_path=f"models/{coin}_lstm.keras"
        cnn.save(cnn_path)
        print(f"  ✅ CNN-LSTM saved → {cnn_path}")
        results["cnn_lstm"]="saved"
    except Exception as e:
        print(f"  ❌ CNN-LSTM failed: {e}")
        results["cnn_lstm"]=f"failed: {e}"

    # ── Train BiLSTM ──────────────────────────────────────────────────────
    print(f"\n  [4/4] Training BiLSTM...")
    try:
        bil = build_bilstm(input_shape)
        bil = compile_model(bil, lr=LR_MAIN, dir_weight=0.4)

        t1=time.time()
        hist2=bil.fit(X_tr,{"price_output":y_tr_p,"direction_output":y_tr_d},
                      epochs=EPOCHS_MAIN,batch_size=BATCH_SIZE,validation_split=0.1,
                      callbacks=callbacks,verbose=0)
        print(f"  BiLSTM phase 1 done in {(time.time()-t1)/60:.1f}min")

        # Fine-tune
        bil=compile_model(bil,lr=LR_FINE,dir_weight=0.8)
        if len(X_ts)>0:
            bil.fit(X_ts,{"price_output":y_ts_p,"direction_output":y_ts_d},
                    epochs=EPOCHS_FINE,batch_size=16,validation_split=0.1,verbose=0)

        bil_path=f"models/{coin}_bilstm.keras"
        bil.save(bil_path)
        print(f"  ✅ BiLSTM saved → {bil_path}")
        results["bilstm"]="saved"
    except Exception as e:
        print(f"  ❌ BiLSTM failed: {e}")
        results["bilstm"]=f"failed: {e}"

    total=round((time.time()-t0)/60,1)
    print(f"\n  {coin} complete in {total} min | CNN-LSTM: {results.get('cnn_lstm')} | BiLSTM: {results.get('bilstm')}")

    # Save metadata in the format expected by incremental_trainer
    # This enables "retrain only new data" logic to work correctly
    try:
        from incremental_trainer import set_last_trained_ts, load_meta, save_meta
        latest_ts = pd.Timestamp(df["Date"].max()) if "Date" in df.columns else pd.Timestamp.now()
        set_last_trained_ts(coin, latest_ts)
        # Also store training config in meta
        meta = load_meta(coin)
        meta.update({
            "last_strategy":      "full_train_both",
            "total_sequences":    len(df),
            "epochs_main":        EPOCHS_MAIN,
            "epochs_fine":        EPOCHS_FINE,
            "latest_market_date": latest_ts.isoformat(),
        })
        save_meta(coin, meta)
        print(f"  Saved meta: last_trained={latest_ts.date()}")
    except Exception as e:
        print(f"  Warning: could not save meta: {e}")

    return {"coin":coin,"status":"ok","time_min":total,**results}


def main():
    coins=sys.argv[1:] if len(sys.argv)>1 else ALL_COINS
    print("="*60)
    print(f"  NeuralPredict — Training CNN-LSTM + BiLSTM")
    print(f"  Coins: {coins}")
    print(f"  Epochs: {EPOCHS_MAIN} main + {EPOCHS_FINE} fine-tune")
    print(f"  Window: {WINDOW_SIZE} | Features: 31")
    print("="*60)

    results=[]
    for c in coins:
        r=train_coin(c)
        results.append(r)

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status=r.get("status","?")
        cnn=r.get("cnn_lstm","—"); bil=r.get("bilstm","—")
        print(f"  {r['coin']:10s}  status={status}  CNN-LSTM={cnn}  BiLSTM={bil}")

    pd.DataFrame(results).to_csv("models/training_summary.csv",index=False)
    print("\n  Summary → models/training_summary.csv")


if __name__ == "__main__":
    main()
