from ultralytics import YOLO

# Pretrained YOLO model.
# You can replace yolo11n.pt with your own pretrained .pt file.
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="license_plate_detector"
)
