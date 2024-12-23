from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StockData(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(10))  # Changed from stock_id to stock
    date = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(10))
    date = Column(DateTime)
    predicted_close = Column(Float)
    predicted_low = Column(Float)
    predicted_high = Column(Float)
    correction = Column(Float, nullable=True)

class EPS(Base):
    __tablename__ = "eps"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(10))
    value = Column(Float)
    change = Column(Float)
    change_percentage = Column(Float)
    year = Column(Integer)

