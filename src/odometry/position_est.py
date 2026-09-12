import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, camera_matrix, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.prev_frame = None
        self.prev_points = None
        self.total_translation = np.zeros(3)
        self.last_known_z = 1.0

    def _derotate_points(self, pts, R):
        K = self.camera_matrix
        pts_norm = cv2.undistortPoints(pts.reshape(-1,1,2), K, None).reshape(-1, 2)
        ones = np.ones((len(pts_norm), 1))
        pts_h = np.hstack([pts_norm, ones])
        R_inv = R.T
        pts_derot = (R_inv @ pts_h.T).T
        pts_derot = pts_derot[:, :2] / (pts_derot[:, 2:3] + 1e-10)
        return pts_derot

    def _to_metric_flow(self, prev_norm, curr_norm_no_rot, z):
        residual = curr_norm_no_rot - prev_norm
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        dx_median = np.median(residual[:, 0])
        dy_median = np.median(residual[:, 1])
        delta_X = -z * dx_median
        delta_Y = -z * dy_median
        return delta_X, delta_Y

    def _estimate_delta_z(self, prev_norm, curr_norm_no_rot):
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        prev_dist = np.sqrt(np.sum(prev_norm ** 2, axis=1))
        curr_dist = np.sqrt(np.sum(curr_norm_no_rot ** 2, axis=1))
        nonzero = (prev_dist > 1e-6) & (curr_dist > 1e-6)
        if np.sum(nonzero) < 10:
            return 0.0
        scale = np.median(curr_dist[nonzero] / prev_dist[nonzero])
        if scale < 0.1 or scale > 10.0:
            return 0.0
        delta_Z = -self.last_known_z * (scale - 1.0)
        return delta_Z

    def estimate_pose(self, prev_pts, curr_pts, z_altitude):
        if prev_pts is None or curr_pts is None or len(prev_pts) < 10 or len(curr_pts) < 10:
            return None, None

        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts,
            self.camera_matrix,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None or E.shape[0] < 3:
            return None, None

        _, R, t_unit, mask = cv2.recoverPose(
            E, curr_pts, prev_pts,
            self.camera_matrix, mask=mask,
        )

        inlier_prev = prev_pts[mask.flatten() == 1]
        inlier_curr = curr_pts[mask.flatten() == 1]
        if len(inlier_prev) < 10:
            return R, np.zeros((3, 1))

        prev_norm = cv2.undistortPoints(inlier_prev.reshape(-1,1,2), self.camera_matrix, None).reshape(-1, 2)
        curr_norm_no_rot = self._derotate_points(inlier_curr, R)

        delta_X, delta_Y = self._to_metric_flow(prev_norm, curr_norm_no_rot, z_altitude)
        delta_Z = self._estimate_delta_z(prev_norm, curr_norm_no_rot)

        t_metric = np.array([[delta_X], [delta_Y], [delta_Z]], dtype=np.float64)

        return R, t_metric

    def estimate_absolute_position(self, R, t):
        if R is None or t is None:
            return self.total_translation
        self.R = R @ self.R
        self.t = self.t + self.R @ t
        self.total_translation = self.t.flatten()
        return self.total_translation

    def set_absolute_position(self, x, y, z):
        self.total_translation = np.array([x, y, z], dtype=np.float64)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.prev_frame = None
        self.prev_points = None
        self.last_known_z = max(z, 0.1)

    def process_frame(self, frame, z_altitude=1.0, gps_health=1):
        if gps_health == 1:
            self.prev_frame = None
            self.prev_points = None
            self.last_known_z = max(z_altitude, 0.1)
            return None

        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            return None

        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            return None

        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, self.prev_points, None, **lk_params)

        if next_points is None:
            self.prev_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            return None

        good_old = self.prev_points[status.flatten() == 1]
        good_new = next_points[status.flatten() == 1]

        if len(good_old) < 8:
            self.prev_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            return None

        R, t_metric = self.estimate_pose(good_old, good_new, z_altitude)

        if R is not None and t_metric is not None:
            position = self.estimate_absolute_position(R, t_metric)
        else:
            position = self.total_translation

        if len(good_new) < 50:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            if new_features is not None:
                new_pts_2d = new_features.reshape(-1, 2)
                existing_2d = good_new.reshape(-1, 2) if good_new.ndim == 3 else good_new
                good_new = np.vstack([existing_2d, new_pts_2d])

        self.prev_frame = frame.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)

        return {
            "position": position.tolist() if isinstance(position, np.ndarray) else position,
            "rotation": R.tolist() if R is not None else None,
            "translation": t_metric.tolist() if t_metric is not None else None,
        }

    def reset(self):
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.prev_frame = None
        self.prev_points = None
        self.total_translation = np.zeros(3)
        self.last_known_z = 1.0
