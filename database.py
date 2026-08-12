import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    logger.error("Chưa cấu hình biến môi trường DB_URL")
    raise ValueError("Thiếu biến môi trường DB_URL")

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Hàm cung cấp phiên làm việc với database cho từng request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
