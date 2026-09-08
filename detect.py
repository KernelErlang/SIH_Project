from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/license_plate_detector/weights/best.pt"
IMAGE_PATH = "test.jpg"

model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

results = model(image, conf=0.4)
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

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Plate {confidence:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

cv2.imwrite("output.jpg", image)
print(f"Detected plates: {count}")
print("Saved: output.jpg")
