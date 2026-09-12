import cv2
import numpy as np


class ORBMatcher:
    def __init__(self, n_features=3000, scale_factor=1.2, n_levels=8, edge_threshold=7, patch_size=19):
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            patchSize=patch_size,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        enhanced = self.clahe.apply(gray)
        keypoints, descriptors = self.orb.detectAndCompute(enhanced, None)
        return keypoints, descriptors

    def match(self, query_desc, train_desc, ratio_threshold=0.75):
        if query_desc is None or train_desc is None:
            return []

        raw_matches = self.bf.knnMatch(query_desc, train_desc, k=2)

        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        return good_matches

    def compute_homography(self, kp1, kp2, matches, min_matches=8, ransac_thresh=4.0):
        if len(matches) < min_matches:
            return None, None, 0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if H is None:
            return None, None, 0

        inliers = int(mask.sum()) if mask is not None else 0
        return H, mask, inliers

    def match_with_verification(self, img1, img2, min_matches=6):
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)

        if des1 is None or des2 is None:
            return None, None, None, 0, "Yetersiz oznitelik"

        matches = self.match(des1, des2)

        if len(matches) < min_matches:
            # Feature matching failed, try template matching
            tm_inliers = self._template_match(img1, img2)
            if tm_inliers >= min_matches:
                return kp1, kp2, matches, tm_inliers, "Basarili"
            return kp1, kp2, matches, 0, "Yetersiz eslesme"

        H, mask, inliers = self.compute_homography(kp1, kp2, matches)

        if inliers >= min_matches:
            return kp1, kp2, matches, inliers, "Basarili"

        # Low inliers from homography, try template match
        tm_inliers = self._template_match(img1, img2)
        if tm_inliers >= min_matches:
            return kp1, kp2, matches, tm_inliers, "Basarili"

        return kp1, kp2, matches, inliers, "Dusuk kalite"

    def template_match(self, img1, img2):
        return self._template_match(img1, img2)

    def _template_match(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        h, w = gray1.shape
        if h < 20 or w < 20:
            return 0
        crop = min(h, w) // 3
        template = gray1[crop:h-crop, crop:w-crop]
        if template.shape[0] < 10 or template.shape[1] < 10:
            return 0
        result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        # Map NCC score [0..1] to pseudo-inliers count
        if max_val > 0.7:
            return int(max_val * 20)
        if max_val > 0.5:
            return int(max_val * 10)
        return 0
