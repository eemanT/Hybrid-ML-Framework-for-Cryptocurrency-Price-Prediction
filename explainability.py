import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np


def shap_explain(model, X_background, X_sample):
    """
    SHAP explanation using KernelExplainer.

    FIX: Uses only 10 background samples (via kmeans) to keep runtime fast.
         Previously used 50 samples which caused 30+ min compute or crash.

    Returns:
        feature_shap: shape (n_features,) — mean contribution per feature
    """
    window     = X_background.shape[1]
    n_features = X_background.shape[2]

    X_bg_flat     = X_background.reshape(len(X_background), -1)
    X_sample_flat = X_sample.reshape(1, -1)

    def predict_flat(X_flat):
        X_3d = X_flat.reshape(-1, window, n_features)
        return model.predict(X_3d, verbose=0).flatten()

    # FIX: kmeans(X, 10) summarizes background into 10 points — much faster
    background = shap.kmeans(X_bg_flat, 10)
    explainer  = shap.KernelExplainer(predict_flat, background)

    # nsamples=50 is fast but still meaningful for FYP
    shap_values_flat     = explainer.shap_values(X_sample_flat, nsamples=50)
    shap_values_reshaped = np.array(shap_values_flat).reshape(window, n_features)

    # Mean absolute SHAP value per feature across all timesteps
    feature_shap = shap_values_reshaped.mean(axis=0)
    return feature_shap


def lime_explain(model, X_background, X_sample, feature_names):
    """
    LIME explanation for LSTM model.
    Flattens 3D input to 2D for tabular LIME.

    Returns:
        LIME Explanation object
    """
    window     = X_background.shape[1]
    n_features = X_background.shape[2]

    X_bg_flat = X_background.reshape(len(X_background), -1)

    # Feature name for each timestep × feature combination
    tiled_names = [
        f"{f}_t-{window - t - 1}"
        for t in range(window)
        for f in feature_names
    ]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bg_flat,
        feature_names=tiled_names,
        mode='regression'
    )

    instance_flat = X_sample.reshape(-1)

    def predict_flat(X_flat):
        X_3d = X_flat.reshape(-1, window, n_features)
        return model.predict(X_3d, verbose=0).flatten()

    exp = explainer.explain_instance(
        instance_flat,
        predict_flat,
        num_features=len(feature_names)
    )
    return exp