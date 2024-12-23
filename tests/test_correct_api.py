import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.db_helper import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base
import os
import pandas as pd
from datetime import datetime

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine and session
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Test client
client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def get_csv_files(year):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and str(year) in f]
    return sorted(csv_files)

def test_correct_api(test_db):
    stock_id = 3  # Assuming ASII has stock_id 3
    year = 2017  # Test for the year 2017
    csv_files = get_csv_files(year)
    
    for csv_file in csv_files:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', csv_file)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Prepare file for upload
        with open(file_path, 'rb') as f:
            response = client.post(
                f"/stock/correct/{stock_id}",
                files={"file": (csv_file, f, "text/csv")}
            )
        
        # Check if the request was successful
        assert response.status_code == 200
        assert response.json() == {"message": "Forecast corrected and updated"}
        
        # Verify predictions were updated
        predictions_response = client.get(f"/stock/predictions/{stock_id}")
        assert predictions_response.status_code == 200
        predictions = predictions_response.json()
        
        # Check if predictions exist and have been updated
        assert len(predictions) > 0
        for prediction in predictions:
            assert prediction['stock_id'] == stock_id
            assert prediction['predicted_close'] is not None
            
            # If correction exists, check if it's reasonable
            if prediction['correction'] is not None:
                assert abs(prediction['correction']) < 1000  # Assuming corrections shouldn't be extremely large
        
        # Optional: Check model performance after each correction
        performance_response = client.get(f"/stock/performance/{stock_id}")
        assert performance_response.status_code == 200
        performance = performance_response.json()
        
        # Log performance metrics
        print(f"Performance after processing {csv_file}:")
        print(f"MAE: {performance['MAE']}")
        print(f"RMSE: {performance['RMSE']}")
        print(f"MAPE: {performance['MAPE']}")
        print("---")

    # After processing all files, check final predictions
    final_predictions_response = client.get(f"/stock/predictions/{stock_id}")
    assert final_predictions_response.status_code == 200
    final_predictions = final_predictions_response.json()
    
    # Verify that we have predictions for future dates
    last_date = max(pred['date'] for pred in final_predictions)
    assert datetime.fromisoformat(last_date) > datetime(year, 12, 31)

if __name__ == "__main__":
    pytest.main([__file__])

