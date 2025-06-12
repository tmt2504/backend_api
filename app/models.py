from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base

class Container(Base):
    __tablename__ = "containers"

    id = Column(Integer, primary_key=True, index=True)
    container_id = Column(String, index=True)
    img_url = Column(String)
    time_process = Column(DateTime)
    engine = Column(String)