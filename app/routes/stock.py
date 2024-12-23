import os
import re
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.utils.db_helper import get_db
from app.services.stock_service import upload_csv, train_model, correct_forecast, get_predictions, model_performance, get_overall_fair_value, calculate_earnings_growth_rate, get_price_range_data
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class PredictionResponse(BaseModel):
    id: int
    stock_id: int
    date: datetime
    predicted_close: float
    correction: float = None

    class Config:
        orm_mode = True

@router.post("/upload/")
async def upload_stock_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await upload_csv(file, db)

@router.post("/train/{stock}")
async def train_stock_model(stock: str, db: Session = Depends(get_db)):
    return await train_model(stock, db)

@router.post("/correct/{stock}")
async def correct_stock_forecast(stock: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await correct_forecast(stock, file, db)

@router.get("/predictions/{stock}")
async def get_stock_predictions(stock: str, db: Session = Depends(get_db)):
    return await get_predictions(stock, db)

@router.get("/performance/{stock}")
async def get_model_performance(stock: str, db: Session = Depends(get_db)):
    return await model_performance(stock, db)

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
    
    results = []
    for csv_file in csv_files:
        file_path = os.path.join(stock_data_dir, csv_file)
        with open(file_path, 'rb') as f:
            file = UploadFile(filename=csv_file, file=f)
            result = await correct_forecast(stock, file, db)
            results.append({"file": csv_file, "result": result})
    
    return {"message": f"Processed {len(results)} files in chronological order", "results": results}


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

