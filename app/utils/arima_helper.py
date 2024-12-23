import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(data):
    # Ensure data is sorted by date
    data = data.sort_values('date')
    
    # Use 'close' column for ARIMA model
    model = ARIMA(data['close'], order=(1,1,1))
    results = model.fit()
    return results

def forecast_next_month(model, steps=4):  # Assuming 4 weeks in a month
    forecast = model.forecast(steps=steps)
    return forecast

def retrain_arima_model(data):
    # Retrain the model with all available data
    new_model = ARIMA(data['close'], order=(1,1,1))
    new_results = new_model.fit()
    return new_results

