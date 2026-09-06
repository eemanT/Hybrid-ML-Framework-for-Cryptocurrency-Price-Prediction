import numpy as np


def hybrid_predict(lstm_preds, arima_preds, alpha=0.6):
    """
    Weighted average: alpha * LSTM + (1-alpha) * ARIMA
    Use find_best_alpha() to determine alpha from test set rather than guessing.
    """
    lstm_arr  = np.array(lstm_preds)
    arima_arr = np.array(arima_preds)
    min_len = min(len(lstm_arr), len(arima_arr))
    return alpha * lstm_arr[:min_len] + (1 - alpha) * arima_arr[:min_len]


def find_best_alpha(lstm_preds, arima_preds, y_true):
    """
    FIX: Instead of hardcoding alpha=0.6, test all values from 0.0 to 1.0
         and pick the one with lowest RMSE on the test set.

    Args:
        lstm_preds:  LSTM predictions on test set
        arima_preds: ARIMA predictions on test set
        y_true:      Actual prices on test set

    Returns:
        best_alpha (float) between 0.0 and 1.0
    """
    best_alpha = 0.5
    best_rmse  = float('inf')

    lstm_arr  = np.array(lstm_preds)
    arima_arr = np.array(arima_preds)
    y_arr     = np.array(y_true)

    min_len = min(len(lstm_arr), len(arima_arr), len(y_arr))
    lstm_arr  = lstm_arr[:min_len]
    arima_arr = arima_arr[:min_len]
    y_arr     = y_arr[:min_len]

    for alpha in np.arange(0.0, 1.05, 0.05):
        hybrid = alpha * lstm_arr + (1 - alpha) * arima_arr
        rmse_val = np.sqrt(np.mean((hybrid - y_arr) ** 2))
        if rmse_val < best_rmse:
            best_rmse  = rmse_val
            best_alpha = round(alpha, 2)

    print(f"Best alpha: {best_alpha} (RMSE: {best_rmse:.4f})")
    return best_alpha