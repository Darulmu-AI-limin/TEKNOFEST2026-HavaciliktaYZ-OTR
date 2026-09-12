import cv2
import numpy as np


def detect_features(gray_frame, max_corners=200, quality_level=0.01, min_distance=10):
    if len(gray_frame.shape) == 3:
        gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_frame.copy()

    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False,
        k=0.04,
    )

    return features


def compute_optical_flow(prev_gray, curr_gray, prev_points, win_size=(21, 21), max_level=3):
    if len(prev_gray.shape) == 3:
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    if len(curr_gray.shape) == 3:
        curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_BGR2GRAY)

    if prev_points is None or len(prev_points) == 0:
        return None, None, None

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1e-4,
    )

    try:
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **lk_params
        )
    except cv2.error:
        return None, None, None

    if next_points is None:
        return None, None, None

    good_old = prev_points[status.flatten() == 1]
    good_new = next_points[status.flatten() == 1]
    good_status = status[status.flatten() == 1]

    return good_old, good_new, good_status
