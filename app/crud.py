from sqlalchemy.orm import Session
from app.models import Container
from datetime import datetime

def get_all_containers(db: Session):
    return db.query(Container).all()

def insert_container(db: Session, container_id: str, img_url: str, time_process: datetime):
    container = Container(container_id=container_id, img_url=img_url, time_process=time_process)
    db.add(container)
    db.commit()
    db.refresh(container)
    return container

def remove_container(db: Session, id: int):
    container = db.query(Container).filter_by(id=id).first()
    if container:
        db.delete(container)
        db.commit()
        return True
    return False
