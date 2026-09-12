import cv2
import numpy as np
from .orb_matcher import ORBMatcher
from .sift_matcher import SIFTMatcher


class AdaptiveMatcher:
    def __init__(self, reference_images=None, mode="hybrid", min_match_threshold=8, quality_threshold=15):
        self.orb = ORBMatcher()
        self.sift = SIFTMatcher()
        self.mode = mode
        self.min_match_threshold = min_match_threshold
        self.quality_threshold = quality_threshold
        self.reference_images = reference_images if reference_images else []
        self.reference_features = self._precompute_features()

    def _precompute_features(self):
        features = []
        for ref in self.reference_images:
            kp_orb, des_orb = self.orb.detect_and_compute(ref["image"])
            kp_sift, des_sift = self.sift.detect_and_compute(ref["image"])
            h, w = ref["image"].shape[:2]
            features.append({
                "id": ref.get("id", 0),
                "name": ref.get("name", "unknown"),
                "orb_kp": kp_orb,
                "orb_des": des_orb,
                "sift_kp": kp_sift,
                "sift_des": des_sift,
                "ref_h": h,
                "ref_w": w,
            })
        return features

    def _preprocess_image(self, image):
        processed = image.copy()
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed.copy()
        equalized = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(equalized)
        return enhanced

    def _match_with_strategy(self, img, ref_feat, strategy="ORB"):
        if strategy == "ORB":
            kp1, des1 = self.orb.detect_and_compute(img)
            if des1 is None or ref_feat["orb_des"] is None:
                return None, None, None, 0, "ORB basarisiz", None
            matches = self.orb.match(des1, ref_feat["orb_des"])
            kp_best = self.orb
            kp2 = ref_feat["orb_kp"]
        elif strategy == "SIFT":
            kp1, des1 = self.sift.detect_and_compute(img)
            if des1 is None or ref_feat["sift_des"] is None:
                return None, None, None, 0, "SIFT basarisiz", None
            matches = self.sift.match(des1, ref_feat["sift_des"])
            kp_best = self.sift
            kp2 = ref_feat["sift_kp"]
        else:
            return None, None, None, 0, "Bilinmeyen strateji", None

        if len(matches) < self.min_match_threshold:
            return kp1, kp2, matches, 0, f"Yetersiz eslesme ({len(matches)})", None

        if len(kp1) == 0 or len(kp2) == 0:
            return kp1, kp2, matches, 0, "Anahtar nokta yok", None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(src_pts) < 4 or len(dst_pts) < 4:
            return kp1, kp2, matches, 0, "Yetersiz nokta", None

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return kp1, kp2, matches, 0, "Homografi basarisiz", None

        inliers = int(mask.sum()) if mask is not None else 0

        bbox = None
        if inliers >= self.min_match_threshold:
            rh = ref_feat.get("ref_h", 0)
            rw = ref_feat.get("ref_w", 0)
            if rh > 0 and rw > 0:
                corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, H)
                xs = projected[:, 0, 0]
                ys = projected[:, 0, 1]
                x1 = int(round(min(xs)))
                y1 = int(round(min(ys)))
                x2 = int(round(max(xs)))
                y2 = int(round(max(ys)))
                bbox = [x1, y1, x2, y2]

        return kp1, kp2, matches, inliers, "Basarili", bbox

    def match(self, image, reference_id=None):
        processed = self._preprocess_image(image)

        results = []
        for ref_feat in self.reference_features:
            if reference_id is not None and ref_feat["id"] != reference_id:
                continue

            bbox = None
            if self.mode in ("hybrid", "ORB"):
                kp1, kp2, matches, inliers, status, bbox = self._match_with_strategy(processed, ref_feat, "ORB")
            else:
                status = "Atlandi"
                inliers = 0
                kp1 = kp2 = matches = None

            if self.mode == "hybrid" and (inliers is None or inliers < self.quality_threshold):
                _, _, _, sift_inliers, sift_status, sift_bbox = self._match_with_strategy(processed, ref_feat, "SIFT")
                if sift_inliers is not None and sift_inliers > (inliers or 0):
                    kp1, kp2, matches, inliers, status, bbox = self._match_with_strategy(processed, ref_feat, "SIFT")

            success = inliers is not None and inliers >= self.min_match_threshold
            results.append({
                "reference_id": ref_feat["id"],
                "reference_name": ref_feat["name"],
                "success": success,
                "inliers": inliers if inliers is not None else 0,
                "total_matches": len(matches) if matches is not None else 0,
                "status": status,
                "bbox": bbox if success and bbox is not None else [0, 0, 0, 0],
                "keypoints_query": kp1,
                "keypoints_train": kp2,
                "matches": matches,
            })

        return results

    def add_reference(self, image, name, ref_id=None):
        if ref_id is None:
            ref_id = len(self.reference_images)
        ref_entry = {"id": ref_id, "name": name, "image": image}
        self.reference_images.append(ref_entry)
        kp_orb, des_orb = self.orb.detect_and_compute(image)
        kp_sift, des_sift = self.sift.detect_and_compute(image)
        h, w = image.shape[:2]
        self.reference_features.append({
            "id": ref_id,
            "name": name,
            "orb_kp": kp_orb,
            "orb_des": des_orb,
            "sift_kp": kp_sift,
            "sift_des": des_sift,
            "ref_h": h,
            "ref_w": w,
        })
        return ref_id


def load_teknofest_references(teknofest_dir="data/teknofest_previous_year"):
    """Load previous-year TEKNOFEST images as reference objects for matching.

    This function loads images from the specified directory (default:
    data/teknofest_previous_year/) and prepares them for use with
    AdaptiveMatcher. The images are expected to be .png or .jpg files
    of objects from previous TEKNOFEST competition sessions.
    """
    import glob
    import os
    refs = []
    if not os.path.exists(teknofest_dir):
        return refs
    for fpath in sorted(glob.glob(os.path.join(teknofest_dir, "*.*g"))):
        img = cv2.imread(fpath)
        if img is not None:
            name = os.path.splitext(os.path.basename(fpath))[0]
            refs.append({"image": img, "name": name, "id": len(refs)})
    return refs
