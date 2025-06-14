import os
import base64
import cv2
import numpy as np
from datetime import datetime

from app import utilities as u
from app.config import OUTPUT_DIR, model

container_parts = {'owner', 'serial', 'dv', 'size'}

def process_img_and_save_to_disk(image_base64: str, time_process: datetime):
    # Remove prefix if present
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]

    # Decode base64 image to OpenCV image
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    image_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    detections = model(image_original)[0]

    # Clone image for each OCR engine
    image_tess = image_original.copy()
    image_trocr = image_original.copy()

    # OCR result containers
    container_tess_parts = {'owner': '', 'serial': '', 'dv': '','size': ''}
    container_trocr_parts = {'owner': '', 'serial': '', 'dv': '', 'size':''}

    for box in detections.boxes:
        cls = int(box.cls.item())
        label = model.names[cls]
        if label not in container_parts:
            continue

        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = u.crop_image_with_box(image_original, xyxy)
        cleaned = u.clean_ocr_image(cropped)

        # Tesseract OCR
        tess = u.tesseract_ocr(cleaned)
        fields_tess = u.extract_fields(tess)
        u.draw_box_and_label(image_tess, xyxy, label, fields_tess)
        container_tess_parts[label] = tess.strip()

        # TrOCR OCR
        trocr = u.trocr_ocr(cleaned)
        fields_trocr = u.extract_fields(trocr)
        u.draw_box_and_label(image_trocr, xyxy, label, fields_trocr)
        container_trocr_parts[label] = trocr.strip()

    # Build container IDs
    container_tess_id = container_tess_parts['owner'] + container_tess_parts['serial'] + container_tess_parts['dv']
    container_trocr_id = container_trocr_parts['owner'] + container_trocr_parts['serial'] + container_trocr_parts['dv']

    time_str = time_process.strftime("%Y%m%d%H%M%S")

    # Save Tesseract image
    filename_tess = f"{time_str}_{container_tess_id}_{container_tess_parts['size']}_tess.jpg"
    save_path_tess = os.path.join(OUTPUT_DIR, filename_tess)
    cv2.imwrite(save_path_tess, image_tess)

    # Save TrOCR image
    filename_trocr = f"{time_str}_{container_trocr_id}_{container_tess_parts['size']}_trocr.jpg"
    save_path_trocr = os.path.join(OUTPUT_DIR, filename_trocr)
    cv2.imwrite(save_path_trocr, image_trocr)

    return {
        "tesseract": {
            "container_id": container_tess_id,
            "image_url": f"/static/output/{filename_tess}"
        },
        "trocr": {
            "container_id": container_trocr_id,
            "image_url": f"/static/output/{filename_trocr}"
        }
    }