from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base

DATABASE_URL = "mysql+pymysql://user:pass@localhost/db_name"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

