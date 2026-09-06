import numpy as np

def forecast_arima(df, steps=5):
    """
    Fit auto_arima on closing prices — automatically finds best (p,d,q) order.

    FIX: Replaced hardcoded ARIMA(5,1,0) with pmdarima auto_arima which
         tests multiple parameter combinations and picks the best one
         based on AIC score. Much more reliable for volatile crypto data.

    Install: pip install pmdarima
    """
    close_prices = df['Close'].dropna().values

    try:
        from pmdarima import auto_arima
        model = auto_arima(
            close_prices,
            seasonal=False,       # crypto has no fixed seasonality
            stepwise=True,        # faster search
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_q=5,
            information_criterion='aic'
        )
        forecast = model.predict(n_periods=steps)
        return np.array(forecast)

    except ImportError:
        # Fallback to basic ARIMA if pmdarima not installed
        print("pmdarima not installed. Run: pip install pmdarima")
        print("Falling back to ARIMA(5,1,0)...")
        from statsmodels.tsa.arima.model import ARIMA
        try:
            model = ARIMA(close_prices, order=(5, 1, 0))
            model_fit = model.fit()
            return np.array(model_fit.forecast(steps=steps))
        except Exception as e:
            print(f"ARIMA failed: {e}. Using last price.")
            return np.full(steps, close_prices[-1])

    except Exception as e:
        print(f"auto_arima failed: {e}. Using last price as fallback.")
        return np.full(steps, close_prices[-1])