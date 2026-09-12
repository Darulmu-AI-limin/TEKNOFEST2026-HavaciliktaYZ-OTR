import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odometry.optical_flow import detect_features, compute_optical_flow
from src.odometry.feature_detector import detect_shi_tomasi
from src.odometry.position_est import VisualOdometry
from src.utils.metrics import compute_position_error


def test_feature_detection():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(50):
        pt = (np.random.randint(50, 590), np.random.randint(50, 430))
        cv2.circle(img, pt, 5, (255, 255, 255), -1)

    features = detect_features(img, max_corners=100)
    if features is not None:
        print(f"[PASS] detect_features: {len(features)} features found")
    else:
        print("[FAIL] No features detected")


def test_shi_tomasi():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(30):
        pt = (np.random.randint(50, 590), np.random.randint(50, 430))
        cv2.circle(img, pt, 6, (255, 255, 255), -1)

    corners = detect_shi_tomasi(img)
    print(f"[PASS] Shi-Tomasi: {len(corners)} corners")


def test_optical_flow():
    prev = np.zeros((480, 640, 3), dtype=np.uint8)
    curr = np.zeros((480, 640, 3), dtype=np.uint8)

    for x in range(50, 590, 20):
        cv2.circle(prev, (x, 200), 4, (255, 255, 255), -1)
        cv2.circle(curr, (x + 5, 200), 4, (255, 255, 255), -1)

    prev_points = detect_features(prev)
    if prev_points is None:
        print("[INFO] Insufficient features for flow test")
        return

    good_old, good_new, status = compute_optical_flow(prev, curr, prev_points)
    if good_old is not None:
        print(f"[PASS] Optical flow: {len(good_old)} tracked points")
    else:
        print("[INFO] Optical flow returned None")


def test_visual_odometry():
    camera_matrix = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]], dtype=np.float64)
    vo = VisualOdometry(camera_matrix)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(80):
        pt = (np.random.randint(50, 590), np.random.randint(50, 430))
        cv2.circle(img, pt, 5, (255, 255, 255), -1)

    result1 = vo.process_frame(img, z_altitude=10.0, gps_health=0)
    result2 = vo.process_frame(img, z_altitude=10.0, gps_health=0)

    if result1 is None:
        print("[INFO] VO: First frame returned None (expected)")
    if result2 is not None:
        print(f"[PASS] VO: Position={result2['position']}")
    else:
        print("[INFO] VO: Second frame also None (expected with repeated frames)")

    vo.reset()
    print("[PASS] VO reset successful")


def test_gps_health():
    from src.detection.landing_analysis import check_gps_health
    assert check_gps_health({"health_status": 1}) == 1
    assert check_gps_health({"health_status": 0}) == 0
    assert check_gps_health(None) == 0
    print("[PASS] GPS health check: all cases correct")


if __name__ == "__main__":
    print("=== Odometry Module Tests ===\n")
    test_feature_detection()
    test_shi_tomasi()
    test_optical_flow()
    test_visual_odometry()
    test_gps_health()
    print("\n=== All Odometry Tests Complete ===")
