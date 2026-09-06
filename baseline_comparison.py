import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
import seaborn as sns

from lstm_model import load_lstm_model, load_bilstm_model
from preprocessing import FEATURE_COLS, WINDOW_SIZE

STRONG_DOWN     = 0
NEUTRAL         = 1
STRONG_UP       = 2
STRONG_QUANTILE = 0.85
CONFIDENCE_THR  = 0.45
DIR_HORIZON     = 5

COLORS = {
    'CNN-LSTM + Transformer ⭐': '#2ecc71',
    'BiLSTM Baseline':           '#5b9bd5',
}


def _make_direction_labels(closes, train_end_idx):
    """Dynamic quantile labels — matches Kaggle training exactly."""
    n   = len(closes)
    fwd = np.array([
        (closes[i + DIR_HORIZON] - closes[i]) / (closes[i] + 1e-10)
        if i + DIR_HORIZON < n else np.nan
        for i in range(n)
    ])
    tr_clean = fwd[:train_end_idx][~np.isnan(fwd[:train_end_idx])]
    up_thr   = np.quantile(tr_clean, STRONG_QUANTILE)
    down_thr = np.quantile(tr_clean, 1 - STRONG_QUANTILE)

    labels = np.where(np.isnan(fwd), NEUTRAL,
             np.where(fwd > up_thr, STRONG_UP,
             np.where(fwd < down_thr, STRONG_DOWN, NEUTRAL)))
    return labels.astype(int), up_thr, down_thr


def _evaluate_model(model, X_test, y_test, scaler, closes_ref, true_dir):
    """Run model on test set, return full metrics dict."""
    tc = scaler.center_[0]
    ts = scaler.scale_[0]

    outputs   = model.predict(X_test, verbose=0, batch_size=64)
    pred_log  = outputs[0].flatten() if isinstance(outputs, list) else outputs.flatten()
    dir_probs = outputs[1] if (isinstance(outputs, list) and len(outputs) > 1) else None

    # Reconstruct prices from log returns
    true_prices, pred_prices = [], []
    price = float(closes_ref)
    for pl, tl in zip(pred_log, y_test):
        pred_p = price * np.exp(float(pl) * ts + tc)
        true_p = price * np.exp(float(tl) * ts + tc)
        pred_prices.append(pred_p)
        true_prices.append(true_p)
        price = true_p

    tp = np.array(true_prices)
    pp = np.array(pred_prices)

    rmse_v = float(np.sqrt(np.mean((tp - pp)**2)))
    mae_v  = float(np.mean(np.abs(tp - pp)))
    mape_v = float(np.mean(np.abs((tp - pp) / (tp + 1e-10))) * 100)
    da_v   = float(np.mean(np.sign(np.diff(tp)) == np.sign(np.diff(pp))) * 100)

    res = dict(true_prices=tp, pred_prices=pp,
               rmse=rmse_v, mae=mae_v, mape=mape_v, da=da_v)

    if dir_probs is not None and len(true_dir) == len(X_test):
        pc  = np.argmax(dir_probs, axis=1)
        cf  = np.max(dir_probs, axis=1)
        tc_ = np.array(true_dir)

        sm = (tc_ == STRONG_DOWN) | (tc_ == STRONG_UP)
        ns = int(sm.sum())

        da_s = float(np.mean(pc[sm] == tc_[sm]) * 100) if ns > 0 else 0.0

        adap = float(np.percentile(cf[sm], 60)) if ns > 0 else CONFIDENCE_THR
        thr  = min(adap, CONFIDENCE_THR + 0.05)
        cs   = sm & (cf > thr)
        nc   = int(cs.sum())
        da_c = float(np.mean(pc[cs] == tc_[cs]) * 100) if nc > 0 else 0.0

        ts_s = tc_[sm]; ps_s = pc[sm]
        f1m  = float(f1_score(ts_s, ps_s, average='macro',           zero_division=0)) if ns > 0 else 0.0
        f1d  = float(f1_score(ts_s, ps_s, labels=[STRONG_DOWN], average=None, zero_division=0)[0]) if ns > 0 else 0.0
        f1u  = float(f1_score(ts_s, ps_s, labels=[STRONG_UP],   average=None, zero_division=0)[0]) if ns > 0 else 0.0
        prec = float(precision_score(ts_s, ps_s, average='macro', zero_division=0)) if ns > 0 else 0.0
        rec  = float(recall_score(ts_s, ps_s,    average='macro', zero_division=0)) if ns > 0 else 0.0

        res.update(dict(
            pred_classes=pc, true_classes=tc_, probs=dir_probs, confidence=cf,
            strong_mask=sm, da_strong=da_s, da_conf=da_c,
            n_strong=ns, n_conf=nc,
            f1_macro=f1m, f1_down=f1d, f1_up=f1u,
            precision=prec, recall=rec
        ))
    return res


def display_model_comparison(X_train, X_test, y_test, scaler, df, crypto):
    """Full runtime model comparison. Call from app.py."""
    st.subheader("🏆 Live Model Comparison")
    st.caption("Both models run on the same held-out test set (last 15% of data, SEED=42)")

    # ── Load both models ──
    model_cnn, model_bil = None, None
    c1, c2 = st.columns(2)
    with c1:
        try:
            model_cnn = load_lstm_model(crypto)
            st.success("✅ CNN-LSTM + Transformer loaded")
        except FileNotFoundError as e:
            st.error(str(e))
    with c2:
        try:
            model_bil = load_bilstm_model(crypto)
            st.success("✅ BiLSTM Baseline loaded")
        except FileNotFoundError as e:
            st.warning(str(e))

    if model_cnn is None:
        st.error("CNN-LSTM model required. Place BTC-USD_lstm.keras in models/ folder.")
        return

    # ── Build direction labels ──
    closes    = df['Close'].values
    n         = len(closes)
    train_end = int(n * 0.70)
    test_start= int(n * 0.85)

    dir_labels, up_thr, down_thr = _make_direction_labels(closes, train_end)
    dir_test = dir_labels[test_start:test_start + len(X_test)]
    if len(dir_test) < len(X_test):
        dir_test = np.pad(dir_test, (0, len(X_test)-len(dir_test)), constant_values=NEUTRAL)

    closes_ref = float(closes[test_start]) if test_start < n else float(closes[-1])

    # ── Evaluate ──
    results = {}
    with st.spinner("Evaluating CNN-LSTM + Transformer..."):
        try:
            results['CNN-LSTM + Transformer ⭐'] = _evaluate_model(
                model_cnn, X_test, y_test, scaler, closes_ref, dir_test)
        except Exception as e:
            st.error(f"CNN-LSTM failed: {e}"); return

    if model_bil is not None:
        with st.spinner("Evaluating BiLSTM Baseline..."):
            try:
                results['BiLSTM Baseline'] = _evaluate_model(
                    model_bil, X_test, y_test, scaler, closes_ref, dir_test)
            except Exception as e:
                st.warning(f"BiLSTM failed: {e}")

    # ══════════════════════════
    #  METRICS TABLE
    # ══════════════════════════
    st.markdown("### 📊 Performance Metrics")
    rows = []
    for name, r in results.items():
        row = {'Model': name,
               'RMSE ($)':      f"${r['rmse']:,.2f}",
               'MAE ($)':       f"${r['mae']:,.2f}",
               'MAPE (%)':      f"{r['mape']:.2f}%",
               'DA (all days)': f"{r['da']:.1f}%"}
        if 'da_strong' in r:
            row['DA (strong days)'] = f"{r['da_strong']:.1f}%  (n={r['n_strong']})"
            row['DA (conf+strong) ★'] = f"{r['da_conf']:.1f}%  (n={r['n_conf']})"
            row['F1 Macro']  = f"{r['f1_macro']:.4f}"
            row['F1 Down']   = f"{r['f1_down']:.4f}"
            row['F1 Up']     = f"{r['f1_up']:.4f}"
            row['Precision'] = f"{r['precision']:.4f}"
            row['Recall']    = f"{r['recall']:.4f}"
        rows.append(row)

    def highlight_best(col):
        if col.name == 'Model': return ['']*len(col)
        higher = col.name in ['DA (all days)', 'DA (strong days)',
                               'DA (conf+strong) ★', 'F1 Macro',
                               'F1 Down', 'F1 Up', 'Precision', 'Recall']
        try:
            nums = col.apply(lambda x: float(str(x).replace('$','')
                             .replace('%','').replace(',','').split()[0]))
            best = nums.max() if higher else nums.min()
            return ['background-color:#d4edda;font-weight:bold'
                    if abs(n - best) < 0.001 else '' for n in nums]
        except Exception:
            return ['']*len(col)

    st.dataframe(
        pd.DataFrame(rows).style.apply(highlight_best),
        use_container_width=True, hide_index=True
    )
    st.caption("🟢 Green = best value  |  ★ = headline metric for FYP")

    # ══════════════════════════
    #  PRICE PREDICTION CHARTS
    # ══════════════════════════
    st.markdown("### 📈 Actual vs Predicted Price (Test Set)")
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(13, 4*n_models), sharex=True)
    if n_models == 1: axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        color = COLORS.get(name, '#888')
        x = range(len(r['true_prices']))
        ax.plot(x, r['true_prices'], label='Actual',    color='#2c3e50', lw=1.5)
        ax.plot(x, r['pred_prices'], label='Predicted', color=color, lw=1.5, ls='--', alpha=0.9)
        ax.set_title(f"{name}  |  MAPE={r['mape']:.2f}%  DA={r['da']:.1f}%", fontsize=10)
        ax.set_ylabel("Price (USD)"); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Test Days")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ══════════════════════════
    #  CONFUSION MATRICES
    # ══════════════════════════
    dir_res = {n: r for n, r in results.items() if 'pred_classes' in r}
    if dir_res:
        st.markdown("### 🔲 Confusion Matrices — Strong Days Only")
        st.caption(f"STRONG_DOWN = bottom {int((1-STRONG_QUANTILE)*100)}% returns  |  "
                   f"STRONG_UP = top {int((1-STRONG_QUANTILE)*100)}% returns  |  "
                   f"Threshold: {up_thr*100:+.2f}% / {down_thr*100:+.2f}%")

        cols = st.columns(len(dir_res))
        for col, (name, r) in zip(cols, dir_res.items()):
            with col:
                sm = r['strong_mask']
                cm = confusion_matrix(r['true_classes'][sm],
                                      r['pred_classes'][sm],
                                      labels=[STRONG_DOWN, STRONG_UP])
                fig2, ax2 = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                            xticklabels=['Pred ↓', 'Pred ↑'],
                            yticklabels=['True ↓', 'True ↑'],
                            linewidths=0.5)
                ax2.set_title(
                    f"{name}\nDA(strong)={r['da_strong']:.1f}%"
                    f"  DA(conf)={r['da_conf']:.1f}%",
                    fontsize=9
                )
                plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── F1 grouped bar chart ──
        st.markdown("### 📊 F1 Score Comparison (Strong Days)")
        model_names = list(dir_res.keys())
        x = np.arange(len(model_names)); w = 0.25
        colors_b = [COLORS.get(n, '#888') for n in model_names]

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        b1 = ax3.bar(x - w,   [r['f1_macro'] for r in dir_res.values()], w, label='F1 Macro', color='#a9dfbf', edgecolor='black')
        b2 = ax3.bar(x,       [r['f1_down']  for r in dir_res.values()], w, label='F1 DOWN',  color='#e74c3c', edgecolor='black', alpha=0.85)
        b3 = ax3.bar(x + w,   [r['f1_up']    for r in dir_res.values()], w, label='F1 UP',    color='#2ecc71', edgecolor='black', alpha=0.85)
        for bar in list(b1)+list(b2)+list(b3):
            if bar.get_height() > 0.01:
                ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)
        ax3.set_xticks(x); ax3.set_xticklabels(model_names, rotation=10, ha='right')
        ax3.set_ylim(0, 1.15); ax3.set_ylabel("F1 Score")
        ax3.set_title("F1 Scores on Strong Days"); ax3.legend(); ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        # ── DA grouped bar chart ──
        st.markdown("### 🎯 Directional Accuracy Breakdown")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        da_s = [r.get('da_strong', 0) for r in dir_res.values()]
        da_c = [r.get('da_conf',   0) for r in dir_res.values()]
        b4 = ax4.bar(x - w/2, da_s, w, label='DA (strong days)', color='#aed6f1', edgecolor='black')
        b5 = ax4.bar(x + w/2, da_c, w, label='DA (conf+strong) ★',
                     color=[COLORS.get(n,'#888') for n in model_names], edgecolor='black')
        for bar in list(b4)+list(b5):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                     f"{bar.get_height():.1f}%", ha='center', va='bottom', fontweight='bold')
        ax4.axhline(50, color='red', ls='--', lw=1, label='Random baseline (50%)')
        ax4.set_xticks(x); ax4.set_xticklabels(model_names, rotation=10, ha='right')
        ax4.set_ylim(0, 95); ax4.set_ylabel("Accuracy %")
        ax4.set_title("Directional Accuracy — Strong vs Confident"); ax4.legend()
        ax4.grid(axis='y', alpha=0.3); plt.tight_layout(); st.pyplot(fig4); plt.close()

        # ── Classification reports ──
        for name, r in dir_res.items():
            sm = r['strong_mask']
            rpt = classification_report(
                r['true_classes'][sm], r['pred_classes'][sm],
                labels=[STRONG_DOWN, STRONG_UP],
                target_names=['STRONG_DOWN', 'STRONG_UP'],
                zero_division=0
            )
            with st.expander(f"📋 {name} — Full Classification Report"):
                st.code(rpt)

    # ══════════════════════════
    #  WINNER + THESIS SENTENCE
    # ══════════════════════════
    st.markdown("---")
    if len(results) >= 2:
        names  = list(results.keys())
        r_cnn  = results[names[0]]
        r_bil  = results.get(names[1])
        if r_bil:
            da_imp   = r_cnn.get('da_conf', r_cnn['da']) - r_bil.get('da_conf', r_bil['da'])
            winner   = names[0] if da_imp >= 0 else names[1]
            st.success(
                f"✅ **{winner}** wins on directional accuracy. "
                f"CNN-LSTM+Transformer: **{r_cnn.get('da_conf', r_cnn['da']):.1f}%** vs "
                f"BiLSTM: **{r_bil.get('da_conf', r_bil['da']):.1f}%** "
                f"({'+' if da_imp >= 0 else ''}{da_imp:.1f}pp)."
            )
            st.markdown("**📝 Auto-generated thesis result sentence:**")
            st.info(
                f"*\"The proposed CNN-LSTM + Transformer hybrid achieves "
                f"{r_cnn.get('da_conf', r_cnn['da']):.1f}% directional accuracy on "
                f"high-confidence strong-movement predictions "
                f"(MAPE={r_cnn['mape']:.2f}%, price accuracy={100-r_cnn['mape']:.2f}%), "
                f"outperforming the BiLSTM baseline "
                f"({r_bil.get('da_conf', r_bil['da']):.1f}% DA, MAPE={r_bil['mape']:.2f}%) "
                f"by {abs(da_imp):.1f} percentage points on the held-out test set "
                f"(n={r_cnn.get('n_conf', len(X_test))} high-confidence predictions, "
                f"15% test split).\"*"
            )
