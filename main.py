from ultralytics import YOLO
import cv2
import torch
import numpy as np
from OCR import PlateCRNN

YOLO_MODEL_PATH = r"C:\Users\HP\runs\detect\license_plate_detector-7\weights\best.pt"
CRNN_MODEL_PATH = r"CRNN MODEL\best_plate_crnn.pt"
IMAGE_PATH = "test.jpg"

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # confirmed from checkpoint['characters']
BLANK_ID = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

device = torch.device("cpu")

# ---- Load YOLO ----
yolo_model = YOLO(YOLO_MODEL_PATH)

# ---- Load CRNN (checkpoint is a dict, not raw weights) ----
crnn_model = PlateCRNN()
checkpoint = torch.load(CRNN_MODEL_PATH, map_location=device)
crnn_model.load_state_dict(checkpoint['model_state_dict'])
crnn_model.eval()


def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (320, 80))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1,1,80,320)
    return tensor


def ctc_greedy_decode(log_probs):
    preds = log_probs.argmax(2).squeeze(1).tolist()
    decoded = []
    prev = None
    for p in preds:
        if p != prev and p != BLANK_ID:
            decoded.append(IDX_TO_CHAR.get(p, "?"))
        prev = p
    return "".join(decoded)


image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

results = yolo_model(image, conf=0.4)
count = 0

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        plate = image[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        count += 1
        cv2.imwrite(f"plate_{count}.jpg", plate)

        plate_tensor = preprocess_plate(plate)
        with torch.no_grad():
            log_probs = crnn_model(plate_tensor)
        plate_text = ctc_greedy_decode(log_probs)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{plate_text} ({confidence:.2f})",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(f"Plate {count}: {plate_text}")

cv2.imwrite("output.jpg", image)
print(f"Detected plates: {count}")
print("Saved: output.jpg")