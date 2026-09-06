import numpy as np


def predict_next_n_days(model, last_window, n_days):
    """
    Autoregressively predict the next n_days prices.
    Compatible with both single-output and dual-output models.

    For dual-output model (price + direction):
      - model.predict() returns [price_array, direction_array]
      - We take index 0 (price output)

    For single-output model:
      - model.predict() returns price_array directly

    Only the Close column (index 0) is updated each step.
    All other features carry forward from the last known step.
    """
    preds    = []
    input_seq = last_window.copy()

    for _ in range(n_days):
        raw = model.predict(input_seq[np.newaxis, :, :], verbose=0)

        # Handle dual-output model
        if isinstance(raw, list):
            pred = float(raw[0][0, 0])   # price_output
        else:
            pred = float(raw[0, 0])

        preds.append(pred)

        # Build new timestep — carry forward features, update Close only
        new_step    = input_seq[-1].copy()
        new_step[0] = pred
        input_seq   = np.vstack([input_seq[1:], new_step])

    return preds
