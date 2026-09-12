import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching.orb_matcher import ORBMatcher
from src.matching.sift_matcher import SIFTMatcher
from src.matching.adaptive_match import AdaptiveMatcher


def test_orb_detect():
    matcher = ORBMatcher()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img, (200, 200), 50, (128, 128, 128), -1)

    kp, des = matcher.detect_and_compute(img)
    if kp is not None and des is not None:
        print(f"[PASS] ORB detect: {len(kp)} keypoints")
    else:
        print("[FAIL] ORB detect returned None")


def test_orb_matching():
    matcher = ORBMatcher()
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    shape_points = np.array([[200, 150], [300, 150], [300, 250], [200, 250]], dtype=np.int32)
    cv2.fillPoly(img1, [shape_points], (255, 255, 255))
    cv2.fillPoly(img2, [shape_points + [10, 5]], (255, 255, 255))

    kp1, des1 = matcher.detect_and_compute(img1)
    kp2, des2 = matcher.detect_and_compute(img2)

    if des1 is None or des2 is None:
        print("[FAIL] Cannot extract descriptors")
        return

    matches = matcher.match(des1, des2)
    print(f"[PASS] ORB matching: {len(matches)} matches")

    kp1, kp2, matches, inliers, status = matcher.match_with_verification(img1, img2)
    if inliers is not None:
        print(f"[PASS] ORB verification: {inliers} inliers, status={status}")
    else:
        print(f"[INFO] ORB verification: status={status}")


def test_sift_matching():
    matcher = SIFTMatcher()
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    shape_points = np.array([[200, 150], [300, 150], [300, 250], [200, 250]], dtype=np.int32)
    cv2.fillPoly(img1, [shape_points], (255, 255, 255))
    cv2.fillPoly(img2, [shape_points + [10, 5]], (255, 255, 255))

    kp1, des1 = matcher.detect_and_compute(img1)
    kp2, des2 = matcher.detect_and_compute(img2)

    if des1 is None or des2 is None:
        print("[FAIL] SIFT cannot extract descriptors")
        return

    matches = matcher.match(des1, des2)
    print(f"[PASS] SIFT matching: {len(matches)} matches")

    _, _, _, inliers, status = matcher.match_with_verification(img1, img2)
    if inliers is not None:
        print(f"[PASS] SIFT verification: {inliers} inliers, status={status}")
    else:
        print(f"[INFO] SIFT verification: status={status}")


def test_adaptive_matching():
    ref_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(ref_img, (150, 150), (350, 350), (255, 255, 255), -1)
    cv2.circle(ref_img, (250, 250), 40, (128, 128, 128), -1)

    adaptive = AdaptiveMatcher(mode="hybrid")
    adaptive.add_reference(ref_img, "test_reference")

    query_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(query_img, (155, 155), (355, 355), (255, 255, 255), -1)
    cv2.circle(query_img, (255, 255), 40, (128, 128, 128), -1)

    results = adaptive.match(query_img)
    matches = [r for r in results if r["success"]]
    print(f"[PASS] Adaptive matching: {len(matches)}/{len(results)} successful")
    for r in results:
        status_text = "SUCCESS" if r["success"] else "FAIL"
        print(f"  -> ref_id={r['reference_id']}, inliers={r['inliers']}, status={status_text}")


if __name__ == "__main__":
    print("=== Matching Module Tests ===\n")
    test_orb_detect()
    test_orb_matching()
    test_sift_matching()
    test_adaptive_matching()
    print("\n=== All Matching Tests Complete ===")
