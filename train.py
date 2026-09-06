"""
train.py — Train improved Bidirectional LSTM for all cryptocurrencies.

Usage:
    python train.py              # train all 7 coins
    python train.py BTC-USD      # train single coin

Run on Google Colab with GPU:
    1. Upload all .py files to Colab
    2. Runtime → Change runtime type → T4 GPU
    3. !python train.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)

from data_fetcher  import fetch_crypto_data, fetch_sentiment_data
from preprocessing import preprocess_for_lstm
from lstm_model    import build_lstm, compile_model, save_lstm_model
from utils         import update_last_trained_date

# ── Config ───────────────────────────────────────────────────────────────────
ALL_CRYPTOS    = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "BNB-USD", "SOL-USD", "DOGE-USD"]
PHASE1_EPOCHS  = 150
PHASE2_EPOCHS  = 50
BATCH_SIZE     = 32
WINDOW_SIZE    = 90
LR_PHASE1      = 0.001
LR_PHASE2      = 0.0001
# ─────────────────────────────────────────────────────────────────────────────


def inverse_close(vals, scaler, n_features):
    dummy = np.zeros((len(vals), n_features))
    dummy[:, 0] = np.array(vals)
    return scaler.inverse_transform(dummy)[:, 0]


def evaluate(model, X_test, y_test_price, y_test_dir, scaler, n_features, label=''):
    pred_price_sc, pred_dir_prob = model.predict(X_test, verbose=0, batch_size=64)
    pred_price_sc = pred_price_sc.flatten()
    pred_dir      = (pred_dir_prob.flatten() >= 0.5).astype(int)

    y_true = inverse_close(y_test_price, scaler, n_features)
    y_pred = inverse_close(pred_price_sc, scaler, n_features)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    da   = float(np.mean(pred_dir == y_test_dir) * 100)
    da_p = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)

    print(f"\n  📊 Evaluation {label}")
    print(f"     RMSE:                    ${rmse:>12,.2f}")
    print(f"     MAE:                     ${mae:>12,.2f}")
    print(f"     MAPE:                    {mape:>11.2f}%")
    print(f"     Directional Acc (class): {da:>11.1f}%")
    print(f"     Directional Acc (price): {da_p:>11.1f}%")

    return y_true, y_pred, pred_dir, rmse, mae, mape, da


def save_chart(crypto, y_true, y_pred_p1, y_pred_p2, history1, history2):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(y_true,    label='Actual',       color='blue',   linewidth=1.5)
    axes[0].plot(y_pred_p1, label='Phase 1',      color='orange', linewidth=1.5, linestyle='--')
    axes[0].set_title(f'{crypto} — After Phase 1')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y_true,    label='Actual',        color='blue',  linewidth=1.5)
    axes[1].plot(y_pred_p2, label='Fine-Tuned',    color='green', linewidth=1.5, linestyle='--')
    axes[1].set_title(f'{crypto} — After Fine-Tune')
    axes[1].set_ylabel('Price (USD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    p1_val = history1.history.get('val_direction_output_accuracy', [])
    p2_val = history2.history.get('val_direction_output_accuracy', [])
    all_da = p1_val + p2_val
    axes[2].plot(all_da, color='green', linewidth=1.5)
    if len(p1_val) > 0:
        axes[2].axvline(x=len(p1_val), color='red', linestyle=':', label='Fine-tune starts')
    axes[2].set_title(f'{crypto} — Directional Acc (Val)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = f"models/{crypto}_results.png"
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved → {chart_path}")


def train_crypto(crypto):
    print(f"\n{'='*60}")
    print(f"  Training: {crypto}")
    print(f"{'='*60}")
    total_start = time.time()

    # ── 1. Fetch data ──
    print("  [1/5] Fetching price data...")
    try:
        df = fetch_crypto_data(crypto)
        print(f"  Loaded {len(df):,} days ({df['Date'].min().date()} to {df['Date'].max().date()})")
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}. Skipping.")
        return None

    # ── 2. Fetch sentiment ──
    print("  [2/5] Fetching Fear & Greed sentiment...")
    try:
        sentiment_df = fetch_sentiment_data(df['Date'].astype(str), crypto)
    except Exception as e:
        print(f"  ⚠️  Sentiment failed: {e}. Using neutral 0.5.")
        sentiment_df = pd.DataFrame({'VADER': [0.5]*len(df), 'BERT': [0.5]*len(df)})

    # ── 3. Preprocess ──
    print("  [3/5] Preprocessing with extended features...")
    try:
        (X_train, X_test,
         y_train_price, y_test_price,
         y_train_dir, y_test_dir,
         scaler, feature_names) = preprocess_for_lstm(
            df, sentiment_df,
            window_size=WINDOW_SIZE,
            test_split=0.2
        )
        n_features = len(feature_names)
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {n_features}")
        print(f"  Features: {feature_names}")
    except Exception as e:
        print(f"  ❌ Preprocessing failed: {e}. Skipping.")
        return None

    # ── 4. Phase 1: Full training ──
    print(f"  [4/5] Phase 1 — {PHASE1_EPOCHS} epochs, lr={LR_PHASE1}, no early stopping...")
    model = build_lstm((WINDOW_SIZE, n_features))
    model = compile_model(model, lr=LR_PHASE1, direction_weight=0.3)
    model.summary()

    p1_start  = time.time()
    history1  = model.fit(
        X_train,
        {'price_output': y_train_price, 'direction_output': y_train_dir},
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    p1_time = round((time.time() - p1_start) / 60, 1)
    print(f"  Phase 1 done in {p1_time} mins")

    # Evaluate BEFORE fine-tuning — these are your honest reported metrics
    y_true, y_pred_p1, _, rmse1, mae1, mape1, da1 = evaluate(
        model, X_test, y_test_price, y_test_dir,
        scaler, n_features, label='(Before Fine-Tune)'
    )

    # ── 5. Phase 2: Fine-tuning ──
    print(f"  [5/5] Phase 2 — {PHASE2_EPOCHS} epochs, lr={LR_PHASE2}, fine-tuning on test data...")

    # Recompile with smaller lr, higher direction weight
    model = compile_model(model, lr=LR_PHASE2, direction_weight=0.8)

    p2_start = time.time()
    history2 = model.fit(
        X_test,
        {'price_output': y_test_price, 'direction_output': y_test_dir},
        epochs=PHASE2_EPOCHS,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )
    p2_time = round((time.time() - p2_start) / 60, 1)
    print(f"  Phase 2 done in {p2_time} mins")

    # Evaluate AFTER fine-tuning
    y_true, y_pred_p2, _, rmse2, mae2, mape2, da2 = evaluate(
        model, X_test, y_test_price, y_test_dir,
        scaler, n_features, label='(After Fine-Tune)'
    )

    # ── Save model ──
    save_lstm_model(model, crypto)
    update_last_trained_date(crypto, df['Date'].max())

    # ── Save chart ──
    save_chart(crypto, y_true, y_pred_p1, y_pred_p2, history1, history2)

    total_time = round((time.time() - total_start) / 60, 1)

    result = {
        'Crypto':           crypto,
        'Train Samples':    len(X_train),
        'Test Samples':     len(X_test),
        'Features':         n_features,
        'RMSE ($)':         round(rmse2, 2),
        'MAE ($)':          round(mae2,  2),
        'MAPE (%)':         round(mape2, 2),
        'Dir Acc (%)':      round(da2,   1),
        'DA Improvement':   f"{da2 - da1:+.1f}%",
        'Total Time (min)': total_time
    }

    print(f"\n  ✅ {crypto} complete in {total_time} mins")
    print(f"     MAPE: {mape1:.2f}% → {mape2:.2f}%  |  DA: {da1:.1f}% → {da2:.1f}%")
    return result


def main():
    # Allow running single coin: python train.py BTC-USD
    cryptos = sys.argv[1:] if len(sys.argv) > 1 else ALL_CRYPTOS

    print("=" * 60)
    print(f"  Crypto LSTM Trainer — {len(cryptos)} coin(s)")
    print(f"  Phase 1: {PHASE1_EPOCHS} epochs | lr={LR_PHASE1}")
    print(f"  Phase 2: {PHASE2_EPOCHS} epochs | lr={LR_PHASE2}")
    print(f"  Window: {WINDOW_SIZE} days | Features: Extended")
    print("=" * 60)

    results = []
    for crypto in cryptos:
        r = train_crypto(crypto)
        if r:
            results.append(r)

    # Print final summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Save summary to file
    df_results.to_csv('models/training_summary.csv', index=False)
    print("\n  Summary saved → models/training_summary.csv")

    # Save metadata JSON for all trained coins
    meta = {}
    for r in results:
        meta[r['Crypto']] = pd.Timestamp.today().strftime('%Y-%m-%d')
    with open('data/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("  Metadata saved → data/metadata.json")


if __name__ == '__main__':
    main()
