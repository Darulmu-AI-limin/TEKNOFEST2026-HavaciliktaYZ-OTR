import requests
import json
import time


class ServerCommunicator:
    def __init__(self, server_url, timeout=10.0, team_name="Darulmu-AI-limin", project_name="BlackScope"):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.team_name = team_name
        self.project_name = project_name
        self.session = requests.Session()
        self._user_url = ""
        self._frame_url_prefix = ""

    def set_user_url(self, url):
        self._user_url = url

    def set_frame_url_prefix(self, prefix):
        self._frame_url_prefix = prefix

    def _get_frame_endpoint(self):
        return f"{self.server_url}/frame"

    def _post_results_endpoint(self):
        return f"{self.server_url}/results"

    def get_frame(self):
        try:
            resp = self.session.get(self._get_frame_endpoint(), timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[SERVER] GET /frame failed: HTTP {resp.status_code}")
                return None
        except requests.RequestException as e:
            print(f"[SERVER] GET /frame error: {e}")
            return None

    def _coerce_bbox_tlbr(self, bbox):
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return int(x1), int(y1), int(x2), int(y2)
        return 0, 0, 0, 0

    def _build_payload(self, frame_id, detections, translations, undefined_objects,
                       user_url="", frame_url=""):
        detected_objects = []
        for det in detections:
            cls = int(det.get("class_id", 0))
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = self._coerce_bbox_tlbr(bbox)

            if cls in (2, 3):
                landing_status = str(det.get("landing_status", 1))
            else:
                landing_status = "-1"

            if cls == 0:
                motion_status = "1" if det.get("is_moving", False) else "0"
            else:
                motion_status = "-1"

            detected_objects.append({
                "cls": str(cls),
                "landing_status": landing_status,
                "motion_status": motion_status,
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2,
            })

        detected_translations = []
        if translations:
            detected_translations.append({
                "translation_x": round(float(translations.get("x", 0)), 4),
                "translation_y": round(float(translations.get("y", 0)), 4),
                "translation_z": round(float(translations.get("z", 0)), 4),
            })

        detected_undefined_objects = []
        if undefined_objects:
            for obj in undefined_objects:
                bbox = obj.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = self._coerce_bbox_tlbr(bbox)
                detected_undefined_objects.append({
                    "object_id": str(obj.get("id", "")),
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2,
                })

        if not user_url:
            user_url = self._user_url
        if not frame_url:
            frame_url = f"{self._frame_url_prefix}{frame_id}"

        payload = {
            "id": f"{self.project_name}_{frame_id}",
            "user": user_url,
            "frame": frame_url,
            "detected_objects": detected_objects,
            "detected_translations": detected_translations,
            "detected_undefined_objects": detected_undefined_objects,
        }

        return payload

    def send_results(self, frame_id, detections, translations=None, undefined_objects=None,
                     user_url="", frame_url=""):
        payload = self._build_payload(frame_id, detections, translations, undefined_objects,
                                      user_url, frame_url)
        try:
            resp = self.session.post(self._post_results_endpoint(), json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return True
            else:
                print(f"[SERVER] POST /results failed: HTTP {resp.status_code}")
                return False
        except requests.RequestException as e:
            print(f"[SERVER] POST /results error: {e}")
            return False

    def send_frame_results(self, pipeline_result):
        frame_id = pipeline_result.get("frame_id", 0)
        detections = pipeline_result.get("detections", [])

        landing = pipeline_result.get("landing_analysis", {})
        uap_landing = {}
        uai_landing = {}
        for item in landing.get("uap_status", []):
            bbox = tuple(item.get("bbox", [0, 0, 0, 0]))
            uap_landing[bbox] = item.get("inis_durumu")
        for item in landing.get("uai_status", []):
            bbox = tuple(item.get("bbox", [0, 0, 0, 0]))
            uai_landing[bbox] = item.get("inis_durumu")

        for det in detections:
            cls = int(det.get("class_id", 0))
            bbox = tuple(det.get("bbox", [0, 0, 0, 0]))
            if cls == 2:
                status = uap_landing.get(bbox)
                if status is not None:
                    det["landing_status"] = 1 if status == 1 else 0
            elif cls == 3:
                status = uai_landing.get(bbox)
                if status is not None:
                    det["landing_status"] = 1 if status == 1 else 0

        pos = pipeline_result.get("position_estimation", None)
        translations = None
        if pos:
            pos_xyz = pos.get("position", [0, 0, 0])
            translations = {"x": pos_xyz[0], "y": pos_xyz[1], "z": pos_xyz[2]}

        matched = pipeline_result.get("matched_objects", [])
        undefined_objects = []
        for m in matched:
            undefined_objects.append({
                "id": m.get("id", ""),
                "bbox": m.get("bbox", [0, 0, 0, 0]),
            })

        return self.send_results(frame_id, detections, translations, undefined_objects)

    def fetch_image(self, image_url):
        try:
            resp = self.session.get(image_url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.content
            else:
                print(f"[SERVER] GET image failed: HTTP {resp.status_code}")
                return None
        except requests.RequestException as e:
            print(f"[SERVER] GET image error: {e}")
            return None

    def close(self):
        self.session.close()
