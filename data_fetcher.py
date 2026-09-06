import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

CRYPTO_START_DATE = {
    "BTC-USD":  "2014-01-01",
    "ETH-USD":  "2015-08-07",
    "XRP-USD":  "2015-08-04",
    "ADA-USD":  "2017-10-01",
    "BNB-USD":  "2017-07-25",
    "SOL-USD":  "2020-03-01",
    "DOGE-USD": "2014-01-01"
}


def fetch_crypto_data(crypto, start_date=None, end_date=None):
    start = start_date or CRYPTO_START_DATE[crypto]
    end   = end_date   or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(crypto, start=start, end=end)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_fear_greed_history():
    try:
        response = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 0, "format": "json"},
            timeout=15
        )
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            print("Fear & Greed API returned empty. Using neutral 0.5.")
            return pd.DataFrame({'Date': [], 'FearGreed': []})
        df = pd.DataFrame(data)
        df['Date'] = (
            pd.to_datetime(df['timestamp'].astype(int), unit='s')
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df['FearGreed'] = df['value'].astype(float) / 100.0
        df = df[['Date', 'FearGreed']].sort_values('Date').reset_index(drop=True)
        print(f"  Fear & Greed: {len(df)} days fetched")
        return df
    except Exception as e:
        print(f"Fear & Greed API failed: {e}. Using neutral 0.5.")
        return pd.DataFrame({'Date': [], 'FearGreed': []})


def get_todays_fear_greed():
    try:
        response = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1},
            timeout=10
        )
        response.raise_for_status()
        value = float(response.json()['data'][0]['value']) / 100.0
        print(f"  Today's Fear & Greed: {value:.2f}")
        return value
    except Exception as e:
        print(f"Could not fetch today's Fear & Greed: {e}. Using 0.5.")
        return 0.5


def fetch_sentiment_data(date_series, crypto_ticker="BTC-USD"):
    date_series = pd.to_datetime(date_series).dt.normalize().dt.tz_localize(None)
    date_series = date_series.reset_index(drop=True)
    fg_df  = fetch_fear_greed_history()
    merged = pd.DataFrame({'Date': date_series})
    if len(fg_df) > 0:
        merged = merged.merge(fg_df, on='Date', how='left')
    else:
        merged['FearGreed'] = 0.5
    merged['FearGreed'] = merged['FearGreed'].fillna(0.5)
    real_count   = (merged['FearGreed'] != 0.5).sum()
    filled_count = (merged['FearGreed'] == 0.5).sum()
    print(f"  Sentiment: {real_count} real F&G days, {filled_count} neutral-filled days")
    return pd.DataFrame({
        'VADER': merged['FearGreed'].values,
        'BERT':  merged['FearGreed'].values
    })
