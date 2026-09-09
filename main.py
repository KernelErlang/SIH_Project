from ultralytics import YOLO
import cv2
import torch
import numpy as np
from OCR import PlateCRNN

YOLO_MODEL_PATH = r"C:\Users\HP\runs\detect\license_plate_detector-7\weights\best.pt"
CRNN_MODEL_PATH = r"CRNN MODEL\best_plate_crnn.pt"   # swap to "demo_finetuned_crnn.pt" if using your fine-tuned version

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_ID = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 320
device = torch.device("cpu")

# ---- Load YOLO ----
yolo_model = YOLO(YOLO_MODEL_PATH)

# ---- Load CRNN ----
crnn_model = PlateCRNN()
checkpoint = torch.load(CRNN_MODEL_PATH, map_location=device)
crnn_model.load_state_dict(checkpoint['model_state_dict'])
crnn_model.eval()


def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def ctc_greedy_decode(log_probs):
    preds = log_probs.argmax(2).squeeze(1).tolist()
    decoded, prev = [], None
    for p in preds:
        if p != prev and p != BLANK_ID:
            decoded.append(IDX_TO_CHAR.get(p, "?"))
        prev = p
    return "".join(decoded)


# ---- Open laptop webcam instead of reading a static image ----
cap = cv2.VideoCapture(0)   # 0 = default laptop camera

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press 'q' to quit, 's' to save the current frame.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = yolo_model(frame, conf=0.4)
    count = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            count += 1

            plate_tensor = preprocess_plate(plate)
            with torch.no_grad():
                log_probs = crnn_model(plate_tensor)
            plate_text = ctc_greedy_decode(log_probs)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{plate_text} ({confidence:.2f})",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("License Plate Detection - Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("captured_frame.jpg", frame)
        print("Saved current frame as captured_frame.jpg")

cap.release()
cv2.destroyAllWindows()