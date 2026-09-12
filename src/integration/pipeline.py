import cv2
import time
import os
import json
import numpy as np

from ..detection.ego_motion import EgoMotionCompensator
from ..detection.landing_analysis import analyze_landing, check_gps_health
from ..detection.train_yolo import detect
from ..detection.preprocessing import preprocess_frame
from ..odometry.position_est import VisualOdometry
from ..matching.adaptive_match import AdaptiveMatcher, load_teknofest_references


class BlackScopePipeline:
    def __init__(self, detection_model, camera_matrix, reference_objects=None, config=None):
        self.detection_model = detection_model
        self.ego_motion = EgoMotionCompensator()
        self.visual_odometry = VisualOdometry(camera_matrix)
        if reference_objects is None or len(reference_objects) == 0:
            teknofest_refs = load_teknofest_references()
            reference_objects = teknofest_refs if teknofest_refs else []
        self.matcher = AdaptiveMatcher(reference_images=reference_objects)
        self.config = config or {}
        self.prev_frame = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []

    def process_frame(self, frame, gps_data=None, landing_zones=None, reference_position=None):
        self.frame_count += 1
        timestamp = time.time()
        frame_start = timestamp

        frame = preprocess_frame(frame)

        result = {
            "frame_id": self.frame_count,
            "timestamp": timestamp,
            "detections": [],
            "landing_analysis": {},
            "position_estimation": None,
            "matched_objects": [],
            "processing_time_ms": 0,
        }

        detections = detect(self.detection_model, frame, conf_threshold=0.25, iou_threshold=0.45)
        result["detections"] = detections

        if self.prev_frame is not None:
            self.ego_motion.detect_motion(self.prev_frame, frame, detections)
        else:
            for det in detections:
                det["is_moving"] = False

        detection_boxes = [d["bbox"] for d in detections if d["class_id"] in (0, 1)]
        if landing_zones:
            result["landing_analysis"] = analyze_landing(detections, landing_zones)

        gps_health = check_gps_health(gps_data)
        z_altitude = gps_data.get("altitude", 1.0) if gps_data else 1.0

        if gps_health == 1 and reference_position is not None:
            rx = reference_position.get("x", 0.0)
            ry = reference_position.get("y", 0.0)
            rz = reference_position.get("z", 0.0)
            self.visual_odometry.set_absolute_position(rx, ry, rz)
            result["position_estimation"] = {
                "position": [rx, ry, rz],
                "method": "reference",
                "gps_health": 1,
            }
        elif gps_health == 0:
            position_result = self.visual_odometry.process_frame(frame, z_altitude, gps_health)
            if position_result:
                result["position_estimation"] = {
                    "position": position_result["position"],
                    "method": "visual_odometry",
                    "gps_health": gps_health,
                }

        if self.matcher.reference_features:
            match_results = self.matcher.match(frame)
            matched = [r for r in match_results if r["success"]]
            result["matched_objects"] = [{"id": r["reference_id"], "name": r["reference_name"], "inliers": r["inliers"], "bbox": r.get("bbox", [0, 0, 0, 0])} for r in matched]

        self.prev_frame = frame.copy()

        frame_end = time.time()
        processing_ms = (frame_end - frame_start) * 1000
        result["processing_time_ms"] = processing_ms

        current_fps = 1.0 / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        return result

    def get_average_fps(self):
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def process_video(self, video_path, gps_data_source=None, landing_zones=None, output_dir=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {fps} FPS, {width}x{height}, {total_frames} frames")

        os.makedirs(output_dir, exist_ok=True) if output_dir else None

        all_results = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gps_data = None
                if gps_data_source and self.frame_count in gps_data_source:
                    gps_data = gps_data_source[self.frame_count]

                result = self.process_frame(frame, gps_data, landing_zones)
                all_results.append(result)

                if self.frame_count % 100 == 0:
                    avg_fps = self.get_average_fps()
                    print(f"Frame {self.frame_count}/{total_frames} - Avg FPS: {avg_fps:.2f}")

        except KeyboardInterrupt:
            print("Processing interrupted.")

        cap.release()

        return all_results

    def save_results_json(self, results, output_path):
        serializable = []
        for r in results:
            s = {k: v for k, v in r.items() if k not in ("keypoints_query", "keypoints_train", "matches")}
            serializable.append(s)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    def reset(self):
        self.ego_motion = EgoMotionCompensator()
        self.visual_odometry.reset()
        self.prev_frame = None
        self.frame_count = 0
        self.fps_history = []
