from ultralytics import YOLO
import os


def train_yolo11m(data_yaml, model_name="yolo11m.pt", epochs=100, imgsz=640, batch=16, project="blackscope_detection"):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="exp",
        patience=20,
        lr0=0.01,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        warmup_epochs=3,
    )
    return model


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)


def detect(model, frame, conf_threshold=0.25, iou_threshold=0.45):
    results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
    detections = []
    if len(results) == 0:
        return detections
    result = results[0]
    if result.boxes is None:
        return detections
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": class_id,
        })
    return detections
