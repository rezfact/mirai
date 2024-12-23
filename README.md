# Mirai Stock Forecasting Project

## Overview

Mirai is an advanced stock forecasting system that uses machine learning techniques, specifically LSTM (Long Short-Term Memory) neural networks, to predict future stock prices. The project is built with Python, FastAPI, and MySQL, providing a robust and scalable solution for stock market analysis and prediction.

## Features

- **Data Management**: Upload and manage stock data via CSV files.
- **LSTM Model**: Utilizes a deep learning LSTM model for accurate stock price predictions.
- **Auto-correction**: Implements an auto-correction mechanism to improve forecast accuracy with new data.
- **API Endpoints**: Provides RESTful API endpoints for data upload, model training, forecasting, and retrieving predictions.
- **Performance Metrics**: Calculates and reports model performance metrics (MSE, RMSE, MAE, MAPE).
- **Prediction Intervals**: Generates lower and upper bounds for price predictions.
- **FastAPI Framework**: Utilizes FastAPI for building high-performance API with automatic interactive documentation.

## Installation

1. Ensure you have the following prerequisites installed:
   - Python 3.8 or higher
   - pip3
   - MySQL 5.7 or higher

2. Clone the repository:
   ```
   git clone https://github.com/rezfact/mirai.git
   cd mirai
   ```

3. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required dependencies:
   ```
   pip3 install -r requirements.txt
   ```

5. Set up the MySQL database:
   - Create a new MySQL database for the project
   - Update the database connection string in `app/utils/db_helper.py`

6. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add the following variables (adjust as needed):
     ```
     DATABASE_URL=mysql+pymysql://username:password@localhost/mirai_db
     SECRET_KEY=your_secret_key_here
     ```

7. Run database migrations:
   ```
   alembic upgrade head
   ```

## Usage

1. Ensure your virtual environment is activated:
   ```
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Start the FastAPI server using uvicorn:
   ```
   uvicorn app.main:app --reload
   ```

3. Access the API documentation at `http://localhost:8000/docs`.

4. Use the provided API endpoints to upload data, train models, and generate predictions.


## Project Structure

```
mirai/
├── app/
│   ├── models/
│   │   └── database.py
│   ├── routes/
│   │   └── stock.py
│   ├── services/
│   │   └── stock_service.py
│   ├── utils/
│   │   ├── arima_helper.py
│   │   └── db_helper.py
│   └── main.py
├── tests/
│   └── test_correct_api.py
├── alembic/
│   └── versions/
├── requirements.txt
├── .env
├── run.py
└── README.md
```

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)

