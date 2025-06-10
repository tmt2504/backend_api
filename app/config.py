import os
import json
import requests
import re
from ultralytics import YOLO

CONFIG_PATH = "../config.json"

# Load config file
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()

# Corrected paths
GITHUB_API_URL = config["github_api_url"]
DEVICE = config["device"]
OUTPUT_DIR = config["output_dir"]
MODEL_FILENAME = config["model_filename"]

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)
VERSION_FILE = os.path.join(WEIGHTS_DIR, "model.version")

# Create folders if needed
for folder in [OUTPUT_DIR, WEIGHTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# GitHub version utils
def get_latest_version_from_github(api_url):
    try:
        r = requests.get(api_url)
        r.raise_for_status()
        items = r.json()
        versions = [item['name'] for item in items if re.match(r'v\d+', item['name'])]
        versions.sort(key=lambda x: int(x[1:]), reverse=True)
        return versions[0] if versions else None
    except Exception as e:
        print(f"Failed to fetch latest version: {e}")
        return None

def get_local_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    return None

def save_local_version(version):
    with open(VERSION_FILE, "w") as f:
        f.write(version)

def download_model_if_new(url, save_path, new_version):
    current_version = get_local_version()
    need_download = (current_version != new_version) or (not os.path.exists(save_path))

    if need_download:
        try:
            print(f"Downloading model from {url} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
            save_local_version(new_version)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
    else:
        print("Model is up to date.")


# Main logic
latest_version = get_latest_version_from_github(GITHUB_API_URL)
if not latest_version:
    raise RuntimeError("Could not determine latest model version.")

RAW_MODEL_URL = f"{config['raw_base_url']}/{latest_version}/weights/{MODEL_FILENAME}"
download_model_if_new(RAW_MODEL_URL, MODEL_PATH, latest_version)

# Load model
model = YOLO(MODEL_PATH)
model.to(DEVICE)
