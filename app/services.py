import os
import base64
import cv2
import utilities as u
from ultralytics import YOLO
import requests
import numpy as np
import json
from datetime import datetime

OUTPUT_DIR = "static/output"
WEIGHTS_DIR = "weights"
CONFIG_PATH = "model_config.json"
MODEL_FILENAME = "best.pt"
MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)
DEVICE = "mps"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
MODEL_URL = config["model_url"]

# Create folders if not exist
for folder in [OUTPUT_DIR, WEIGHTS_DIR]:
    os.makedirs(folder, exist_ok=True)

def download_model(url, save_path):
    if not os.path.exists(save_path):
        try:
            print(f"Downloading model from {url} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
    else:
        print("Model already exists.")

# Download model
download_model(MODEL_URL, MODEL_PATH)
# Load model YOLO
model = YOLO(MODEL_PATH)
model.to(DEVICE)

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

