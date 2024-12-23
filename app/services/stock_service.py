import os
import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.models.database import StockData, Prediction
from datetime import timedelta
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import subprocess
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Disable GPU usage to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calculate_earnings_growth_rate(eps_data):
    """Calculate average growth rate from change_percentage of last 5 years"""
    eps_data = eps_data.sort_values('year', ascending=False)
    historical_data = eps_data.iloc[1:6]  # Skip current year, get next 5 years
    
    avg_growth_rate = historical_data['change_percentage'].mean()
    rounded_growth_rate = round(avg_growth_rate)

    if rounded_growth_rate < 5:
        rounded_growth_rate = 5
    elif rounded_growth_rate > 25:
        rounded_growth_rate = 25

    print(f"Debug: Average growth rate before rounding and constraints: {avg_growth_rate:.2f}%")
    print(f"Debug: Final growth rate after rounding and constraints: {rounded_growth_rate}%")
    
    return rounded_growth_rate

def calculate_peter_lynch_fair_value(eps_data):
    """Calculate fair value using growth rate × current year EPS"""
    growth_rate = calculate_earnings_growth_rate(eps_data)
    
    latest_eps = eps_data.loc[eps_data['year'].idxmax(), 'value']
    
    if latest_eps < 0:
        print("Warning: TTM EPS is negative. Peter Lynch Fair Value result may be unreliable.")
    
    fair_value = growth_rate * latest_eps
    
    print(f"Debug: growth_rate={growth_rate}%, latest_eps={latest_eps}, fair_value={fair_value:.2f}")
    
    return round(fair_value, 2), growth_rate, latest_eps

async def fetch_google_finance_data(stock: str):
    """Fetch latest stock data from Google Finance"""
    try:
        result = subprocess.run(['node', 'fetch_google_finance_data.js', stock], 
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching Google Finance data: {e}")
        return None

async def get_overall_fair_value(stock: str, db: Session):
    """Get fair value calculation for a stock"""
    # Get last 6 years of data (current year + 5 historical years)
    eps_data = db.query(EPS).filter(EPS.stock == stock).order_by(EPS.year.desc()).limit(6).all()
    df = pd.DataFrame([d.__dict__ for d in eps_data])
    
    if df.empty:
        print(f"Warning: No EPS data found for stock {stock}")
        return 0, 0, 0, 0, 0, ""
    
    if len(df) < 6:
        print(f"Warning: Less than 6 years of EPS data available for stock {stock}")
    
    fair_value, growth_rate, latest_eps = calculate_peter_lynch_fair_value(df)
    
    # Get the latest close value from stock_data
    latest_stock_data = db.query(StockData).filter(StockData.stock == stock).order_by(StockData.date.desc()).first()
    current_price = latest_stock_data.close if latest_stock_data else 0
    
    # Calculate upside percentage
    upside_percentage = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
    
    if fair_value == 0:
        print(f"Warning: Calculated overall fair value is 0. Data: {df.to_dict()}")
    
    description = ("Calculated using the average growth rate of net income/earnings over the last 5 years, "
                   "rounded to the nearest integer and constrained between 5% and 25%. "
                   "The fair value is the product of the growth rate and the latest EPS.")
    
    return fair_value, growth_rate, latest_eps, current_price, upside_percentage, description

async def get_price_range_data(stock: str, db: Session):
    """Get yearly lowest and highest prices with percentage changes"""
    stock_data = db.query(StockData).filter(StockData.stock == stock).all()
    df = pd.DataFrame([d.__dict__ for d in stock_data])
    
    if df.empty:
        return []
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    yearly_data = df.groupby('year').agg({
        'low': 'min',
        'high': 'max'
    }).reset_index()
    
    yearly_data['low_change'] = yearly_data['low'].pct_change() * 100
    yearly_data['high_change'] = yearly_data['high'].pct_change() * 100
    
    result = []
    for _, row in yearly_data.iterrows():
        year = int(row['year'])
        lowest_price = row['low']
        highest_price = row['high']
        low_change = row['low_change']
        high_change = row['high_change']
        
        low_change_str = f"{low_change:.1f}%" if not pd.isna(low_change) else "N/A"
        high_change_str = f"{high_change:.1f}%" if not pd.isna(high_change) else "N/A"
        
        result.append({
            "year": year,
            "lowest_price": lowest_price,
            "highest_price": highest_price,
            "low_change": low_change_str,
            "high_change": high_change_str
        })
    
    result.sort(key=lambda x: x['year'], reverse=True)
    
    return result

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(5)  # Output all 5 features
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def generate_weekly_dates(start_date, num_weeks):
    return [start_date + timedelta(weeks=i) for i in range(num_weeks)]

async def upload_csv(file: UploadFile, db: Session):
    df = pd.read_csv(file.file)
    for _, row in df.iterrows():
        stock_data = StockData(
            stock=row['ID'],
            date=pd.to_datetime(row['Date']),
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume']
        )
        db.add(stock_data)
    db.commit()
    return {"message": "Data uploaded successfully"}

async def train_model(stock: str, db: Session):
    data = db.query(StockData).filter(StockData.stock == stock).order_by(StockData.date).all()
    df = pd.DataFrame([d.__dict__ for d in data])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    features = ['open', 'high', 'low', 'close', 'volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    if len(scaled_data) < 10:
        # Fallback method for very small datasets
        last_price = df['close'].iloc[-1]
        future_dates = generate_weekly_dates(df.index[-1] + timedelta(days=1), 4)
        
        # Delete existing predictions for this stock
        db.query(Prediction).filter(Prediction.stock == stock).delete()
        
        for date in future_dates:
            prediction = Prediction(
                stock=stock,
                date=date,
                predicted_close=last_price,
                predicted_low=last_price * 0.95,
                predicted_high=last_price * 1.05
            )
            db.add(prediction)
        
        db.commit()
        return {"message": "Not enough data for LSTM model. Simple prediction based on last price generated."}
    
    seq_length = min(10, len(scaled_data) // 2)
    X, y = create_sequences(scaled_data, seq_length)
    
    if len(X) < 2:
        raise ValueError("Not enough data for training. Need at least 2 sequences.")
    
    # Adjust validation split based on data size
    validation_split = min(0.1, (len(X) - 1) / len(X))
    
    model = build_lstm_model((seq_length, len(features)))
    model.fit(X, y, epochs=50, batch_size=min(32, len(X)), validation_split=validation_split, verbose=0)
    
    last_sequence = X[-1].reshape(1, seq_length, len(features))
    last_date = df.index[-1]
    future_dates = generate_weekly_dates(last_date + timedelta(days=1), 4)
    future_predictions = []
    
    for _ in range(4):  # Predict for 4 weeks
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0])
        
        # Update last_sequence for the next iteration
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, :] = next_pred

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Use the standard deviation of the entire dataset for prediction intervals
    std_dev = np.std(df['close'])
    lower_bound = future_predictions[:, 3] - 1.96 * std_dev
    upper_bound = future_predictions[:, 3] + 1.96 * std_dev
    
    # Delete existing predictions for this stock
    db.query(Prediction).filter(Prediction.stock == stock).delete()
    
    for date, pred, lower, upper in zip(future_dates, future_predictions[:, 3], lower_bound, upper_bound):
        prediction = Prediction(
            stock=stock,
            date=date,
            predicted_close=pred,
            predicted_low=lower,
            predicted_high=upper
        )
        db.add(prediction)
    
    db.commit()
    return {"message": "Model trained and predictions generated successfully"}


async def correct_forecast(stock: str, file: UploadFile, db: Session):
    try:
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        new_data.columns = new_data.columns.str.lower()
        new_data['date'] = pd.to_datetime(new_data['date'])
        new_data = new_data.sort_values('date')
        
        existing_data = db.query(StockData).filter(StockData.stock == stock).order_by(StockData.date).all()
        existing_df = pd.DataFrame([d.__dict__ for d in existing_data])
        
        if not existing_df.empty:
            existing_df['date'] = pd.to_datetime(existing_df['date'])
        
        combined_data = pd.concat([existing_df, new_data]).drop_duplicates(subset=['date'], keep='last').sort_values('date')
        combined_data.set_index('date', inplace=True)
        
        for _, row in new_data.iterrows():
            stock_data = StockData(
                stock=stock,
                date=row['date'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            db.merge(stock_data)
        
        features = ['open', 'high', 'low', 'close', 'volume']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(combined_data[features])
        
        # Adjust sequence length based on data size
        seq_length = min(5, len(scaled_data) // 4)
        
        if len(scaled_data) < seq_length + 1:
            # Fallback method for very small datasets
            last_price = combined_data['close'].iloc[-1]
            future_dates = generate_weekly_dates(combined_data.index[-1] + timedelta(days=1), 4)
            for date in future_dates:
                prediction = Prediction(
                    stock=stock,
                    date=date,
                    predicted_close=last_price,
                    predicted_low=last_price * 0.95,
                    predicted_high=last_price * 1.05,
                    correction=None
                )
                db.add(prediction)
            db.commit()
            return {"message": "Not enough data for LSTM model. Simple prediction based on last price generated."}
        
        X, y = create_sequences(scaled_data, seq_length)
        
        # Use all data for training if there's not enough for validation
        if len(X) < 2:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            # Adjust train-test split ratio for smaller datasets
            test_size = max(0.1, min(0.2, 1 - (seq_length + 1) / len(X)))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        model = build_lstm_model((seq_length, len(features)))
        
        # Adjust batch size and validation split based on data size
        batch_size = min(32, len(X_train))
        validation_split = 0.1 if len(X_train) > 10 else 0
        
        model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=validation_split, verbose=0)
        
        all_sequences = create_sequences(scaled_data, seq_length)[0]
        predictions = model.predict(all_sequences)
        predicted_close = scaler.inverse_transform(predictions)[:, 3]  # Close price is at index 3
        
        std_dev = np.std(combined_data['close'])
        lower_bound = predicted_close - 1.96 * std_dev
        upper_bound = predicted_close + 1.96 * std_dev
        
        # Get the last prediction date
        last_prediction = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date.desc()).first()
        last_prediction_date = last_prediction.date if last_prediction else combined_data.index[0]
        
        # Update existing predictions and add new ones
        for date, pred, lower, upper in zip(combined_data.index[seq_length:], predicted_close, lower_bound, upper_bound):
            if date <= last_prediction_date:
                continue  # Skip dates that already have predictions
            
            existing_prediction = db.query(Prediction).filter(
                Prediction.stock == stock,
                Prediction.date == date
            ).first()
            
            if existing_prediction:
                existing_prediction.predicted_close = pred
                existing_prediction.predicted_low = lower
                existing_prediction.predicted_high = upper
            else:
                new_prediction = Prediction(
                    stock=stock,
                    date=date,
                    predicted_close=pred,
                    predicted_low=lower,
                    predicted_high=upper,
                    correction=None
                )
                db.add(new_prediction)
        
        # Apply exponential smoothing for correction
        alpha = 0.3  # Smoothing factor
        for date in combined_data.index[seq_length:]:
            if date <= last_prediction_date:
                continue  # Skip dates that already have predictions
            
            actual_data = combined_data.loc[date]
            prediction = db.query(Prediction).filter(
                Prediction.stock == stock,
                Prediction.date == date
            ).first()
            
            if prediction and not pd.isna(actual_data['close']):
                correction = actual_data['close'] - prediction.predicted_close
                smoothed_correction = alpha * correction + (1 - alpha) * (prediction.correction if prediction.correction is not None else 0)
                prediction.correction = smoothed_correction
                prediction.predicted_close += smoothed_correction
                prediction.predicted_low += smoothed_correction
                prediction.predicted_high += smoothed_correction
        
        # Generate new predictions for the next 4 weeks
        last_sequence = all_sequences[-1]
        future_dates = generate_weekly_dates(combined_data.index[-1] + timedelta(days=1), 4)
        future_predictions = []
        
        for _ in range(4):
            next_pred = model.predict(np.array([last_sequence]))
            future_predictions.append(next_pred[0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_pred[0]
        
        future_predictions = np.array(future_predictions)
        future_predictions = scaler.inverse_transform(future_predictions)[:, 3]  # Close price is at index 3
        future_lower_bound = future_predictions - 1.96 * std_dev
        future_upper_bound = future_predictions + 1.96 * std_dev
        
        for date, pred, lower, upper in zip(future_dates, future_predictions, future_lower_bound, future_upper_bound):
            new_prediction = Prediction(
                stock=stock,
                date=date,
                predicted_close=pred,
                predicted_low=lower,
                predicted_high=upper,
                correction=None
            )
            db.add(new_prediction)
        
        db.commit()
        return {"message": "Forecast corrected and updated for all dates, including new predictions for the next 4 weeks"}
    
    except ValueError as ve:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_predictions(stock: str, db: Session):
    predictions = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date).all()
    return predictions

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

async def model_performance(stock: str, db: Session):
    data = db.query(StockData).filter(StockData.stock == stock).order_by(StockData.date).all()
    predictions = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date).all()
    
    df = pd.DataFrame([d.__dict__ for d in data])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    pred_df = pd.DataFrame([p.__dict__ for p in predictions])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df.set_index('date', inplace=True)
    
    merged_df = df.join(pred_df[['predicted_close']], how='inner')
    merged_df.dropna(inplace=True)
    
    performance = evaluate_model(merged_df['close'], merged_df['predicted_close'])
    return performance

