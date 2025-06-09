import os
import json
import requests
from ultralytics import YOLO

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
