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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
        results = s.process_img_and_save_to_disk(req.image_base64, time_process)

        tess_result = results["tesseract"]
        container_tess = crud.insert_container(
            db,
            tess_result["container_id"],
            tess_result["image_url"],
            tess_result["iso_code"],
            time_process,
            engine="tesseract"
        )

        trocr_result = results["trocr"]
        container_trocr = crud.insert_container(
            db,
            trocr_result["container_id"],
            trocr_result["image_url"],
            trocr_result["iso_code"],
            time_process,
            engine="trocr"
        )

        return {
            "tesseract": {
                "container_id": container_tess.container_id,
                "image_url": container_tess.img_url,
                "iso_code": container_tess.iso_code,
                "time_process": container_tess.time_process.isoformat()
            },
            "trocr": {
                "container_id": container_trocr.container_id,
                "image_url": container_trocr.img_url,
                "iso_code": container_trocr.iso_code,
                "time_process": container_trocr.time_process.isoformat()
            }
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
