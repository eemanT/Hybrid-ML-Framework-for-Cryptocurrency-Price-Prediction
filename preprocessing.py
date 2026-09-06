import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df):
    df = df.copy()

    # Basic indicators
    df['SMA_10']        = df['Close'].rolling(window=10).mean()
    df['SMA_30']        = df['Close'].rolling(window=30).mean()
    df['Daily_Return']  = df['Close'].pct_change()
    df['Volatility']    = df['Close'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Momentum']      = df['Close'].diff(10)

    # RSI
    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands — strong directional signal at extremes
    bb_mid         = df['Close'].rolling(20).mean()
    bb_std         = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (bb_mid + 1e-10)
    df['BB_pos']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)

    # MACD — reliable directional momentum indicator
    ema12           = df['Close'].ewm(span=12, adjust=False).mean()
    ema26           = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']      = ema12 - ema26
    df['MACD_sig']  = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_sig']

    # OBV — volume confirms price direction
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV']        = obv
    df['OBV_change'] = df['OBV'].pct_change()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_for_lstm(df, sentiment_df=None, window_size=90, test_split=0.2):
    """
    Preprocess data for improved Bidirectional LSTM.
    Returns X, y_price, y_direction for dual-output training.
    y_direction: 1=price went up, 0=price went down (used for classification head)
    """
    df = df.copy()
    df = add_technical_indicators(df)

    if sentiment_df is not None:
        sentiment_df = sentiment_df.reset_index(drop=True)
        min_len      = min(len(df), len(sentiment_df))
        df           = df.iloc[-min_len:].reset_index(drop=True)
        sentiment_df = sentiment_df.iloc[-min_len:].reset_index(drop=True)
        df['VADER']  = sentiment_df['VADER'].values
        df['BERT']   = sentiment_df['BERT'].values
        feature_cols = [
            'Close', 'SMA_10', 'SMA_30', 'RSI',
            'Daily_Return', 'Volatility', 'Volume_Change', 'Momentum',
            'BB_upper', 'BB_lower', 'BB_width', 'BB_pos',
            'MACD', 'MACD_sig', 'MACD_hist',
            'OBV_change', 'VADER', 'BERT'
        ]
    else:
        feature_cols = [
            'Close', 'SMA_10', 'SMA_30', 'RSI',
            'Daily_Return', 'Volatility', 'Volume_Change', 'Momentum',
            'BB_upper', 'BB_lower', 'BB_width', 'BB_pos',
            'MACD', 'MACD_sig', 'MACD_hist', 'OBV_change'
        ]

    features  = df[feature_cols].values
    n         = len(features)
    split_idx = int(n * (1 - test_split)) if test_split > 0 else n

    features_train = features[:split_idx]
    features_test  = features[split_idx:]

    # Fit scaler only on train — no leakage
    scaler               = MinMaxScaler()
    train_scaled         = scaler.fit_transform(features_train)
    test_scaled          = scaler.transform(features_test) if len(features_test) > 0 else np.array([])

    def make_sequences(data):
        X, y_price, y_dir = [], [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i])
            y_price.append(data[i, 0])
            # Direction: 1 if price went up from previous step, 0 if down
            direction = 1 if data[i, 0] > data[i - 1, 0] else 0
            y_dir.append(direction)
        return np.array(X), np.array(y_price), np.array(y_dir)

    X_train, y_train_price, y_train_dir = make_sequences(train_scaled)

    if len(test_scaled) > window_size:
        combined_test                       = np.concatenate([train_scaled[-window_size:], test_scaled], axis=0)
        X_test, y_test_price, y_test_dir    = make_sequences(combined_test)
    else:
        X_test       = np.array([]).reshape(0, window_size, len(feature_cols))
        y_test_price = np.array([])
        y_test_dir   = np.array([])

    return (X_train, X_test,
            y_train_price, y_test_price,
            y_train_dir, y_test_dir,
            scaler, feature_cols)
