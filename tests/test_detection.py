import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.train_yolo import load_model, detect
from src.detection.ego_motion import EgoMotionCompensator
from src.detection.landing_analysis import analyze_landing
from src.utils.metrics import calculate_iou


def test_model_load():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base, "models", "yolo11m_visdrone.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base, "yolo11m.pt")
    if not os.path.exists(model_path):
        print("[INFO] No model file found, using ultralytics YOLO11m")
        try:
            from ultralytics import YOLO
            m = YOLO("yolo11m.pt")
            print(f"[PASS] Model loaded via ultralytics: yolo11m.pt")
            return m
        except Exception as e:
            print(f"[FAIL] Model load error: {e}")
            return None
    try:
        model = load_model(model_path)
        print(f"[PASS] Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[FAIL] Model load error: {e}")
        return None


def test_detect(model):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    try:
        detections = detect(model, img)
        print(f"[PASS] detect() returned {len(detections)} detections")
    except Exception as e:
        print(f"[FAIL] detect() error: {e}")


def test_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    box3 = [20, 20, 30, 30]

    iou1 = calculate_iou(box1, box2)
    iou2 = calculate_iou(box1, box3)

    assert iou1 > 0, "Overlapping boxes should have IoU > 0"
    assert iou2 == 0, "Non-overlapping boxes should have IoU = 0"
    print(f"[PASS] IoU tests: overlapping={iou1:.2f}, non-overlapping={iou2:.2f}")


def test_ego_motion():
    compensator = EgoMotionCompensator()
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame1, (100, 100), 30, (255, 255, 255), -1)
    cv2.circle(frame2, (110, 100), 30, (255, 255, 255), -1)

    H = compensator.compute_homography(frame1, frame2)
    if H is not None:
        print(f"[PASS] Homography computed: shape={H.shape}")
    else:
        print("[INFO] Homography returned None (low texture)")
    moving = compensator.detect_motion(frame1, frame2, [{"class_id": 0, "bbox": [0, 0, 100, 100]}])



def test_landing_analysis():
    detections = [
        {"bbox": [50, 50, 150, 150], "confidence": 0.9, "class_id": 2},
        {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class_id": 0},
        {"bbox": [400, 400, 500, 500], "confidence": 0.7, "class_id": 3},
    ]
    landing_zones = [
        {"bbox": [100, 100, 200, 200], "type": "UAP"},
    ]
    result = analyze_landing(detections, landing_zones)
    assert "uap_status" in result, "UAP analysis missing"
    assert "uai_status" in result, "UAI analysis missing"
    print(f"[PASS] Landing analysis: {len(result['uap_status'])} UAP, {len(result['uai_status'])} UAI")


if __name__ == "__main__":
    print("=== Detection Module Tests ===\n")
    model = test_model_load()
    if model:
        test_detect(model)
    test_iou()
    test_ego_motion()
    test_landing_analysis()
    print("\n=== All Detection Tests Complete ===")
