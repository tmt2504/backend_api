import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Xác định thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trỏ đến thư mục database nằm cùng cấp backend_api/
DB_PATH = os.path.join(BASE_DIR, "../database/containers.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # Tạo thư mục nếu chưa có

DATABASE_URL = f"sqlite:///{DB_PATH}"  # Đường dẫn tuyệt đối

# Tạo engine và session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
