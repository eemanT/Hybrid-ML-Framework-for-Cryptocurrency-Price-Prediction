from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """
    FIX: Added MAPE (Mean Absolute Percentage Error).
    More interpretable than RMSE — tells you % error in plain English.
    e.g. MAPE=3.2% means predictions are off by 3.2% on average.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    """Measures how often the model correctly predicts price direction (up/down)."""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)