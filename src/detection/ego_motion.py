import cv2
import numpy as np


class EgoMotionCompensator:
    def __init__(self, motion_threshold=8.0, min_matches=12):
        self.motion_threshold = motion_threshold
        self.min_matches = min_matches
        self.prev_gray = None
        self.prev_detections = []
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compute_homography(self, prev_frame, curr_frame):
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        kp_prev, des_prev = self.orb.detectAndCompute(gray_prev, None)
        kp_curr, des_curr = self.orb.detectAndCompute(gray_curr, None)
        if des_prev is None or des_curr is None or len(kp_prev) < self.min_matches or len(kp_curr) < self.min_matches:
            return None
        matches = self.bf.match(des_prev, des_curr)
        if len(matches) < self.min_matches:
            return None
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        src = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return H

    def detect_motion(self, prev_frame, curr_frame, curr_detections):
        H = self.compute_homography(prev_frame, curr_frame)

        vehicle_dets = [d for d in curr_detections if d.get("class_id") == 0]
        for d in curr_detections:
            if d.get("class_id") != 0:
                d["is_moving"] = False

        if len(vehicle_dets) == 0:
            return []

        if H is None:
            for d in vehicle_dets:
                d["is_moving"] = False
            return []

        H_inv = np.linalg.inv(H)
        h, w = curr_frame.shape[:2]
        moving_indices = []

        for idx, det in enumerate(curr_detections):
            box = det.get("bbox", [0, 0, 0, 0])
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            center = np.float32([[[cx, cy]]])

            warped_prev = cv2.perspectiveTransform(center, H_inv)
            wx, wy = warped_prev[0, 0]

            if wx < 0 or wx >= w or wy < 0 or wy >= h:
                det["is_moving"] = False
                continue

            crop_size = 16
            x1_c = max(0, int(cx) - crop_size)
            y1_c = max(0, int(cy) - crop_size)
            x2_c = min(w, int(cx) + crop_size)
            y2_c = min(h, int(cy) + crop_size)

            wx1_p = max(0, int(wx) - crop_size)
            wy1_p = max(0, int(wy) - crop_size)
            wx2_p = min(w, int(wx) + crop_size)
            wy2_p = min(h, int(wy) + crop_size)

            if (x2_c - x1_c) < 8 or (y2_c - y1_c) < 8 or (wx2_p - wx1_p) < 8 or (wy2_p - wy1_p) < 8:
                det["is_moving"] = False
                continue

            gray_pre = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
            gray_cur = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

            patch_curr = gray_cur[y1_c:y2_c, x1_c:x2_c]
            patch_prev_warped = gray_pre[wy1_p:wy2_p, wx1_p:wx2_p]

            if patch_curr.shape != patch_prev_warped.shape:
                det["is_moving"] = False
                continue

            diff = cv2.absdiff(patch_curr, patch_prev_warped)
            mean_diff = float(diff.mean())
            det["is_moving"] = mean_diff > self.motion_threshold
            if det["is_moving"]:
                moving_indices.append(idx)

        return moving_indices
