from datetime import datetime
import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import crud
from app.models import Base, Container
from app.database import SessionLocal, engine
from app import services as s

Base.metadata.create_all(bind=engine)
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ContainerRequest(BaseModel):
    image_base64: str

@app.get("/get_containers")
def get_containers(db: Session = Depends(get_db)):
    return crud.get_all_containers(db)


@app.post("/insert_container")
def insert_container(req: ContainerRequest, db: Session = Depends(get_db)):
    try:
        time_process = datetime.utcnow()
        container_id, img_url = s.process_img_and_save_to_disk(req.image_base64, time_process)
        container = crud.insert_container(db, container_id, img_url, time_process)
        return {
            "container_id": container.container_id,
            "image_url": container.img_url,
            "time_process": container.time_process.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove_container")
def remove_container(id: int, db: Session = Depends(get_db)):
    container = db.query(Container).filter_by(id=id).first()
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")

    file_path = container.img_url.lstrip("/")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"[INFO] Deleted image file: {file_path}")
        except Exception as e:
            print(f"[WARN] Could not delete image file {file_path}: {e}")

    success = crud.remove_container(db, id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete container in DB")

    return {"message": "Container and image removed"}



if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
