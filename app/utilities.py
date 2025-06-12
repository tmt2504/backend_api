import cv2
import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Common
def crop_image_with_box(img, box, padding=5):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)
    return img[y1:y2, x1:x2]

def draw_box_and_label(image, xyxy, label, text):
    x1, y1, x2, y2 = xyxy
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text_lines = [f"{label}: {text}"]

    y_offset = y1 - 10
    for line in text_lines:
        cv2.putText(image, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset -= 20  # mỗi dòng cách nhau 20px

def clean_ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding and noise reduction
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_fields(text):
    return text if text else ""

# TesseractOCR
def tesseract_ocr(image):
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return pytesseract.image_to_string(image, config=config).strip()

# TrOCR
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
trocr_model.eval()

def trocr_ocr(image):
    # Convert to RGB PIL Image
    if image.ndim == 2:  # grayscale
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image

    pil_img = Image.fromarray(img_rgb)

    # Preprocess
    inputs = processor(images=pil_img, return_tensors="pt") # type: ignore
    with torch.no_grad():
        generated_ids = trocr_model.generate(**inputs) # type: ignore

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # type: ignore
    return generated_text.strip()