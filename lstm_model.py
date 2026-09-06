import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
import os

MODEL_DIR = "models/"


def build_lstm(input_shape):
    """
    Improved Bidirectional LSTM with Attention mechanism.
    Two outputs:
      - price_output:     regression  (predicts next scaled close price)
      - direction_output: classification (predicts up=1 / down=0)

    Training both outputs simultaneously forces the model to learn
    directional patterns explicitly alongside price levels.
    """
    inputs = Input(shape=input_shape)

    # Bidirectional LSTM layer 1 — reads sequence forward and backward
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)

    # Bidirectional LSTM layer 2
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Attention — learns which timesteps matter most
    attention_out = Attention()([x, x])
    pooled        = GlobalAveragePooling1D()(attention_out)

    # Shared dense
    shared = Dense(64, activation='relu')(pooled)
    shared = Dropout(0.2)(shared)

    # Output 1: Price regression
    price = Dense(32, activation='relu')(shared)
    price = Dense(1, name='price_output')(price)

    # Output 2: Direction classification
    direction = Dense(32, activation='relu')(shared)
    direction = Dense(1, activation='sigmoid', name='direction_output')(direction)

    model = Model(inputs=inputs, outputs=[price, direction])
    return model


def compile_model(model, lr=0.001, direction_weight=0.3):
    """
    Compile with dual loss.
    direction_weight: how much to weight direction loss vs price loss.
    Increase during fine-tuning to push harder on directional accuracy.
    """
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={
            'price_output':     'mse',
            'direction_output': 'binary_crossentropy'
        },
        loss_weights={
            'price_output':     1.0,
            'direction_output': direction_weight
        },
        metrics={
            'price_output':     'mae',
            'direction_output': 'accuracy'
        }
    )
    return model


def save_lstm_model(model, crypto):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = f"{MODEL_DIR}{crypto}_lstm.keras"
    model.save(path)
    print(f"  Model saved → {path}")


def load_lstm_model(crypto):
    path_keras = f"{MODEL_DIR}{crypto}_lstm.keras"
    path_h5    = f"{MODEL_DIR}{crypto}_lstm.h5"
    if os.path.exists(path_keras):
        return load_model(path_keras)
    elif os.path.exists(path_h5):
        return load_model(path_h5)
    raise FileNotFoundError(f"No saved model for {crypto}. Train first.")
