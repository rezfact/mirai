from sqlalchemy import Column, Integer, Float, DateTime, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StockData(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(10))
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

class EPS(Base):
    __tablename__ = "eps"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(10))
    value = Column(Float)
    change = Column(Float)
    change_percentage = Column(Float)
    year = Column(Integer)

class StockInfo(Base):
    __tablename__ = "stock_info"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50))
    event_subtype = Column(String(50))
    stock_name = Column(String(100))
    security_code = Column(String(10))
    security_name = Column(String(100))
    record_date = Column(DateTime)
    effective_date = Column(DateTime)
    start_date = Column(DateTime, nullable=True)
    deadline_date = Column(DateTime, nullable=True)
    ca_description = Column(Text)
    crawl_date = Column(DateTime)
    event_date = Column(DateTime)  # New column for the date from the URL