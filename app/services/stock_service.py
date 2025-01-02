import os
import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.models.database import StockData, Prediction
from datetime import timedelta
from typing import List
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_weekly_dates(start_date, num_weeks):
    return [start_date + timedelta(weeks=i) for i in range(num_weeks)]

def simple_moving_average(data, window):
    return data['close'].rolling(window=min(window, len(data))).mean().iloc[-1]

def exponential_moving_average(data, span):
    return data['close'].ewm(span=span, adjust=False).mean().iloc[-1]

async def upload_csv(stock: str, file_content: bytes, db: Session):
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        df['Date'] = pd.to_datetime(df['Date'])
        
        rows_added = 0
        rows_updated = 0
        
        for _, row in df.iterrows():
            existing_data = db.query(StockData).filter(
                StockData.stock == stock,
                StockData.date == row['Date']
            ).first()
            
            if existing_data:
                existing_data.open = float(row['Open'])
                existing_data.high = float(row['High'])
                existing_data.low = float(row['Low'])
                existing_data.close = float(row['Close'])
                existing_data.volume = float(row['Volume'])
                rows_updated += 1
            else:
                stock_data = StockData(
                    stock=stock,
                    date=row['Date'],
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume'])
                )
                db.add(stock_data)
                rows_added += 1
        
        db.commit()
        logger.info(f"Successfully uploaded CSV for stock {stock}. Rows added: {rows_added}, Rows updated: {rows_updated}")
        return {"rows_added": rows_added, "rows_updated": rows_updated}
    except Exception as e:
        logger.error(f"Error in upload_csv for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading CSV: {str(e)}")

async def train_model(stock: str, db: Session, prediction_weeks: int = 52):  # Changed to 52 weeks (1 year)
    try:
        data = db.query(StockData).filter(StockData.stock == stock).order_by(StockData.date).all()
        if not data:
            logger.warning(f"No data available for training stock {stock}")
            return {"message": "No data available for training"}

        df = pd.DataFrame([d.__dict__ for d in data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        if len(df) < 4:
            logger.warning(f"Insufficient data for any prediction for stock {stock}. Current data points: {len(df)}")
            return {"message": f"Insufficient data for prediction. At least 4 data points are required. Current data points: {len(df)}"}

        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Generate predictions for the next year
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_weeks, freq='W-MON')
        
        if len(df) < 30:
            logger.info(f"Using simple moving average for stock {stock} due to limited data points: {len(df)}")
            # Use simple moving average for small datasets
            last_close = df['close'].iloc[-1]
            sma_5 = simple_moving_average(df, 5)
            sma_all = df['close'].mean()
            
            predictions = []
            for i in range(prediction_weeks):
                weight = max(0, 1 - i*0.05)  # Decrease weight more slowly for further predictions
                pred = (last_close * 0.5 + sma_5 * 0.3 + sma_all * 0.2) * weight + sma_all * (1 - weight)
                predictions.append(pred)
            
            std_dev = np.std(df['close'])
            lower_bound = [pred - 1.96 * std_dev for pred in predictions]
            upper_bound = [pred + 1.96 * std_dev for pred in predictions]
        else:
            logger.info(f"Training LSTM model for stock {stock} with {len(df)} data points")
            
            # Feature engineering
            df['returns'] = df['close'].pct_change()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            df = df.dropna()
            
            if len(df) == 0:
                logger.warning(f"No valid data points after feature engineering for stock {stock}")
                return {"message": "No valid data points after feature engineering"}
            
            features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'ma_50', 'ma_200', 'ema_20', 'volatility']
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[features])
            
            seq_length = min(30, len(scaled_data) - 1)  # Ensure seq_length is not larger than available data
            X, y = create_sequences(scaled_data, seq_length)
            
            if len(X) == 0:
                logger.warning(f"No sequences could be created for stock {stock}. Available data points: {len(df)}")
                return {"message": f"Insufficient data for sequence creation. Available data points: {len(df)}"}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = build_lstm_model((seq_length, len(features)))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            batch_size = min(32, len(X_train))
            
            model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)
            
            # Evaluate the model
            performance = evaluate_model(model, X_test, y_test, scaler)
            
            last_sequence = X[-1].reshape(1, seq_length, len(features))
            predictions = []
            
            for _ in range(prediction_weeks):
                next_pred = model.predict(last_sequence)
                predictions.append(next_pred[0][3])  # Append only the close price prediction
                
                # Update the last sequence for the next prediction
                new_row = next_pred[0]
                new_row[5] = (new_row[3] - last_sequence[0, -1, 3]) / last_sequence[0, -1, 3]  # Calculate returns
                new_row[6] = np.mean(last_sequence[0, -49:, 3])  # Update MA 50
                new_row[7] = np.mean(last_sequence[0, -199:, 3])  # Update MA 200
                new_row[8] = new_row[3] * 0.0952 + last_sequence[0, -1, 8] * 0.9048  # Update EMA 20
                new_row[9] = np.std(last_sequence[0, -19:, 5])  # Update volatility
                
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, :] = new_row

            predictions = np.array(predictions)
            predictions = scaler.inverse_transform(np.column_stack((np.zeros((len(predictions), 3)), predictions, np.zeros((len(predictions), 6)))))[:, 3]
            
            std_dev = np.std(df['close'])
            lower_bound = predictions - 1.96 * std_dev
            upper_bound = predictions + 1.96 * std_dev

        # Update existing predictions and add new ones
        existing_predictions = db.query(Prediction).filter(Prediction.stock == stock).all()
        existing_dates = {pred.date.date() for pred in existing_predictions}

        for date, pred, lower, upper in zip(future_dates, predictions, lower_bound, upper_bound):
            date = date.date()  # Convert to date object for comparison
            if date in existing_dates:
                # Update existing prediction
                existing_pred = next(p for p in existing_predictions if p.date.date() == date)
                existing_pred.predicted_close = float(pred)
                existing_pred.predicted_low = float(lower)
                existing_pred.predicted_high = float(upper)
            else:
                # Add new prediction
                new_prediction = Prediction(
                    stock=stock,
                    date=date,
                    predicted_close=float(pred),
                    predicted_low=float(lower),
                    predicted_high=float(upper)
                )
                db.add(new_prediction)

        db.commit()
        logger.info(f"Successfully trained model and updated predictions for stock {stock}")
        return {"message": "Model trained and predictions updated successfully", "performance": performance if len(df) >= 30 else None, "data_points": len(df)}
    except Exception as e:
        logger.error(f"Error in train_model for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while training the model: {str(e)}")
    
async def trains(stock: str, csv_files: List[str], db: Session):
    try:
        base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        stock_data_dir = os.path.join(base_data_dir, stock)
        
        if not os.path.exists(stock_data_dir):
            logger.warning(f"No data directory found for stock {stock}")
            raise HTTPException(status_code=404, detail=f"No data directory found for stock {stock}")
        
        if not csv_files:
            logger.warning(f"No CSV files provided for stock {stock}")
            raise HTTPException(status_code=404, detail=f"No CSV files provided for stock {stock}")
        
        results = []
        
        for i, csv_file in enumerate(csv_files):
            file_path = os.path.join(stock_data_dir, csv_file)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"Processing file {i+1}/{len(csv_files)}: {csv_file}")
            
            try:
                # Read CSV file
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Upload CSV data
                upload_result = await upload_csv(stock, file_content, db)
                logger.info(f"Upload result for {csv_file}: {upload_result}")
                
                # Train model and update predictions for the next year (52 weeks)
                logger.info(f"Starting model training for {csv_file}")
                train_result = await train_model(stock, db, prediction_weeks=52)
                logger.info(f"Training result for {csv_file}: {train_result}")
                
                results.append({
                    "file": csv_file,
                    "upload_result": upload_result,
                    "train_result": train_result
                })
            except Exception as e:
                logger.error(f"Error processing file {csv_file}: {str(e)}")
                results.append({
                    "file": csv_file,
                    "error": str(e)
                })
            
            logger.info(f"Completed processing file {i+1}/{len(csv_files)}: {csv_file}")
        
        # Get final prediction count
        final_predictions = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date).all()
        final_prediction_count = len(final_predictions)
        
        # Get date range of predictions
        if final_predictions:
            prediction_start = final_predictions[0].date
            prediction_end = final_predictions[-1].date
        else:
            prediction_start = prediction_end = None
        
        logger.info(f"Successfully processed {len(results)} files for stock {stock}")
        return {
            "message": f"Processed {len(results)} files",
            "results": results,
            "final_prediction_count": final_prediction_count,
            "prediction_date_range": {
                "start": prediction_start.isoformat() if prediction_start else None,
                "end": prediction_end.isoformat() if prediction_end else None
            }
        }
    except Exception as e:
        logger.error(f"Error in trains for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing files: {str(e)}")

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
        seq_length = min(30, len(scaled_data) // 10)
        
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
                    predicted_high=last_price * 1.05
                )
                db.add(prediction)
            db.commit()
            return {"message": "Not enough data for LSTM model. Simple prediction based on last price generated."}
        
        X, y = create_sequences(scaled_data, seq_length)
        
        # Use all data for training if there's not enough for validation
        if len(X) < 100:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            # Adjust train-test split ratio for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = build_lstm_model((seq_length, len(features)))
        
        # Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Adjust batch size and validation split based on data size
        batch_size = min(64, len(X_train))
        validation_split = 0.2 if len(X_train) > 100 else 0
        
        model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], verbose=0)
        
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
                    predicted_high=upper
                )
                db.add(new_prediction)
        
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
                predicted_high=upper
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


def remove_duplicate_predictions(stock: str, db: Session):
    predictions = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date).all()
    unique_dates = set()
    for prediction in predictions:
        if prediction.date not in unique_dates:
            unique_dates.add(prediction.date)
        else:
            db.delete(prediction)
    db.commit()

async def get_predictions(stock: str, db: Session):
    predictions = db.query(Prediction).filter(Prediction.stock == stock).order_by(Prediction.date).all()
    return [p.__dict__ for p in predictions]

# Add this function to evaluate the model's performance
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(np.column_stack((np.zeros((len(y_pred), 3)), y_pred[:,3], np.zeros((len(y_pred), 6)))))[:, 3]
    y_test = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test), 3)), y_test[:,3], np.zeros((len(y_test), 6)))))[:, 3]
    mse = np.mean(np.square(y_pred - y_test))
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse}

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

