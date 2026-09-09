from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt") 

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        name="license_plate_detector",
        device=0
    )