import os
import base64
import cv2
import numpy as np
from datetime import datetime

from app import utilities as u
from app.config import OUTPUT_DIR, model

def process_img_and_save_to_disk(image_base64: str, time_process: datetime):
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]

    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = model(image)[0]
    container_id = ''

    for box in detections.boxes:
        cls = int(box.cls.item())
        label = model.names[cls]
        if label not in ['owner', 'serial', 'dv']:
            continue

        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = u.crop_image_with_box(image, xyxy)
        cleaned = u.clean_ocr_image(cropped)
        text = u.extract_text(cleaned)
        fields = u.extract_fields(text)
        u.draw_box_and_label(image, xyxy, label, fields)

        container_id += text

    # Save output image
    time_str = time_process.strftime("%Y%m%d%H%M%S")
    filename = f"{time_str}_{container_id}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, image)

    return container_id, f"/static/output/{filename}"
