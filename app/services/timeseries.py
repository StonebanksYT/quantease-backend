import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def analyze_timeseries(data):
    series = pd.Series(data['series'])

    # ARIMA model
    model = ARIMA(series, order=(5, 1, 0))  # Example order (p, d, q)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=data.get('forecast_steps', 10))
    return {"forecast": forecast.tolist()}
