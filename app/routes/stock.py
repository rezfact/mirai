import os
import re
import io
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from app.utils.db_helper import get_db
from app.services.stock_service import (
    upload_csv, train_model, correct_forecast, get_predictions, 
    model_performance, get_overall_fair_value, get_price_range_data, trains
)
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class PredictionResponse(BaseModel):
    id: int
    stock_id: int
    date: datetime
    predicted_close: float
    correction: float = None

    class Config:
        orm_mode = True

@router.post("/upload/{stock}")
async def upload_stock_data(stock: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_content = await file.read()
        result = await upload_csv(stock, file_content, db)
        return result
    except Exception as e:
        logger.error(f"Error in upload_stock_data for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the file: {str(e)}")

@router.post("/train/{stock}")
async def train_stock_model(stock: str, db: Session = Depends(get_db)):
    try:
        return await train_model(stock, db)
    except Exception as e:
        logger.error(f"Error in train_stock_model for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while training the model: {str(e)}")

@router.post("/trains/{stock}")
async def auto_correct(stock: str, db: Session = Depends(get_db)):
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    stock_data_dir = os.path.join(base_data_dir, stock)

    if not os.path.exists(stock_data_dir):
        raise HTTPException(status_code=404, detail=f"No data directory found for stock {stock}")

    csv_files = [
        f for f in os.listdir(stock_data_dir)
        if f.endswith('.csv') and f.startswith(f"{stock}-")
    ]

    # Sort by year and month
    csv_files.sort(key=lambda x: (
        int(re.search(r'-(\d{2})-(\d{4})\.csv$', x).group(2)),  # Year
        int(re.search(r'-(\d{2})-(\d{4})\.csv$', x).group(1))   # Month
    ))

    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found for stock {stock}")

    try:
        result = await trains(stock, csv_files, db)
        return result
    except HTTPException as e:
        logger.error(f"HTTP exception in train_multiple_files for stock {stock}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in train_multiple_files for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/correct/{stock}")
async def correct_stock_forecast(stock: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await correct_forecast(stock, file, db)

@router.get("/predictions/{stock}")
async def get_stock_predictions(stock: str, db: Session = Depends(get_db)):
    try:
        predictions = await get_predictions(stock, db)
        return predictions
    except Exception as e:
        logger.error(f"Error in get_stock_predictions for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving predictions: {str(e)}")

@router.get("/performance/{stock}")
async def get_model_performance(stock: str, db: Session = Depends(get_db)):
    try:
        performance = await model_performance(stock, db)
        return performance
    except Exception as e:
        logger.error(f"Error in get_model_performance for stock {stock}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving model performance: {str(e)}")

@router.post("/auto-correct/{stock}")
async def auto_correct(stock: str, db: Session = Depends(get_db)):
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    stock_data_dir = os.path.join(base_data_dir, stock)

    if not os.path.exists(stock_data_dir):
        raise HTTPException(status_code=404, detail=f"No data directory found for stock {stock}")

    csv_files = [
        f for f in os.listdir(stock_data_dir)
        if f.endswith('.csv') and f.startswith(f"{stock}-")
    ]

    # Sort by year and month
    csv_files.sort(key=lambda x: (
        int(re.search(r'-(\d{2})-(\d{4})\.csv$', x).group(2)),  # Year
        int(re.search(r'-(\d{2})-(\d{4})\.csv$', x).group(1))   # Month
    ))

    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found for stock {stock}")
    
    accumulated_data = pd.DataFrame()
    results = []

    for csv_file in csv_files:
        file_path = os.path.join(stock_data_dir, csv_file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
            new_data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            accumulated_data = pd.concat([accumulated_data, new_data]).drop_duplicates().reset_index(drop=True)

        if len(accumulated_data) >= 10:
            csv_content = accumulated_data.to_csv(index=False).encode()
            file = UploadFile(filename=csv_file, file=io.BytesIO(csv_content))
            result = await correct_forecast(stock, file, db)
            results.append({"file": csv_file, "result": result})
        else:
            results.append({"file": csv_file, "result": {"message": f"Accumulating data. Current data points: {len(accumulated_data)}"}})

    return {"message": f"Processed {len(csv_files)} files in chronological order", "results": results}

@router.get("/fair-value/{stock}")
async def overall_fair_value(stock: str, db: Session = Depends(get_db)):
    try:
        fair_value, growth_rate, latest_eps, current_price, upside_percentage, description = await get_overall_fair_value(stock, db)
        
        if fair_value == 0 and growth_rate == 0 and latest_eps == 0:
            raise HTTPException(status_code=404, detail=f"No EPS data found for stock {stock}")
        
        response = {
            "stock": stock,
            "fair_value": fair_value,
            "current": current_price,
            "upside": f"{upside_percentage:.1f}%",
            "method": "Peter Lynch",
            "growth_rate": growth_rate,
            "latest_eps": latest_eps,
            "description": description,
            "warning": "If the TTM EPS is negative, the result may be unreliable."
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/price-range/{stock}")
async def price_range(stock: str, db: Session = Depends(get_db)):
    try:
        price_data = await get_price_range_data(stock, db)
        
        if not price_data:
            raise HTTPException(status_code=404, detail=f"No price data found for stock {stock}")
        
        response = {
            "stock": stock,
            "price_range_data": price_data
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")