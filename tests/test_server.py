import sys, os, json, unittest
import unittest.mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.integration.server_comm import ServerCommunicator


class TestServerCommunicator(unittest.TestCase):
    def setUp(self):
        self.comm = ServerCommunicator("http://test.local:5000")

    def test_build_payload_include_all_fields(self):
        detections = [
            {"class_id": 0, "bbox": [10, 20, 100, 200], "is_moving": True},
            {"class_id": 1, "bbox": [30, 40, 80, 120], "is_moving": False},
            {"class_id": 2, "bbox": [50, 60, 150, 180], "landing_status": 1},
            {"class_id": 3, "bbox": [70, 80, 130, 170], "landing_status": 0},
        ]
        translations = {"x": 1.234, "y": 2.345, "z": 3.456}
        undefined = [{"id": "obj_1", "bbox": [100, 100, 200, 200]}]

        payload = self.comm._build_payload(42, detections, translations, undefined,
                                           user_url="/users/test", frame_url="/frames/42")

        self.assertEqual(payload["id"], "BlackScope_42")
        self.assertEqual(payload["user"], "/users/test")
        self.assertEqual(payload["frame"], "/frames/42")

        objs = payload["detected_objects"]
        self.assertEqual(len(objs), 4)

        self.assertEqual(objs[0]["cls"], "0")
        self.assertEqual(objs[0]["motion_status"], "1")
        self.assertEqual(objs[0]["landing_status"], "-1")
        self.assertEqual(objs[0]["top_left_x"], 10)
        self.assertEqual(objs[0]["top_left_y"], 20)
        self.assertEqual(objs[0]["bottom_right_x"], 100)
        self.assertEqual(objs[0]["bottom_right_y"], 200)

        self.assertEqual(objs[1]["cls"], "1")
        self.assertEqual(objs[1]["motion_status"], "-1")
        self.assertEqual(objs[1]["landing_status"], "-1")

        self.assertEqual(objs[2]["cls"], "2")
        self.assertEqual(objs[2]["motion_status"], "-1")
        self.assertEqual(objs[2]["landing_status"], "1")

        self.assertEqual(objs[3]["cls"], "3")
        self.assertEqual(objs[3]["motion_status"], "-1")
        self.assertEqual(objs[3]["landing_status"], "0")

    def test_build_payload_no_optional(self):
        payload = self.comm._build_payload(1, [], None, None)
        self.assertEqual(payload["detected_objects"], [])
        self.assertEqual(payload["detected_translations"], [])
        self.assertEqual(payload["detected_undefined_objects"], [])

    def test_build_payload_bbox_order(self):
        dets = [{"class_id": 0, "bbox": [200, 300, 100, 150]},]
        payload = self.comm._build_payload(1, dets, None, None)
        obj = payload["detected_objects"][0]
        self.assertEqual(obj["top_left_x"], 100)
        self.assertEqual(obj["top_left_y"], 150)
        self.assertEqual(obj["bottom_right_x"], 200)
        self.assertEqual(obj["bottom_right_y"], 300)

    def test_send_frame_results_structure(self):
        result = {
            "frame_id": 5,
            "detections": [
                {"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9},
                {"class_id": 2, "bbox": [20, 20, 50, 50], "confidence": 0.8},
            ],
            "landing_analysis": {
                "uap_status": [
                    {"bbox": [20, 20, 50, 50], "confidence": 0.8, "inis_durumu": 1, "engel_var": False},
                ],
                "uai_status": [],
            },
            "position_estimation": None,
            "matched_objects": [],
            "processing_time_ms": 15.0,
        }

        with unittest.mock.patch.object(self.comm, 'send_results', return_value=True) as mock:
            self.comm.send_frame_results(result)
            mock.assert_called_once()
            args = mock.call_args
            self.assertEqual(args[0][0], 5)

    def test_landing_status_mapping(self):
        result = {
            "frame_id": 1,
            "detections": [
                {"class_id": 2, "bbox": [0, 0, 10, 10], "confidence": 0.9},
            ],
            "landing_analysis": {
                "uap_status": [
                    {"bbox": [0, 0, 10, 10], "confidence": 0.9, "inis_durumu": 1, "engel_var": False},
                ],
                "uai_status": [],
            },
            "position_estimation": None,
            "matched_objects": [],
            "processing_time_ms": 5.0,
        }

        with unittest.mock.patch.object(self.comm, 'send_results', return_value=True) as mock:
            self.comm.send_frame_results(result)
            dets = mock.call_args[0][1]
            for d in dets:
                if d["class_id"] == 2:
                    self.assertEqual(d["landing_status"], 1)

    def test_inis_uygun_degil_mapping(self):
        result = {
            "frame_id": 1,
            "detections": [
                {"class_id": 3, "bbox": [0, 0, 10, 10], "confidence": 0.9},
            ],
            "landing_analysis": {
                "uap_status": [],
                "uai_status": [
                    {"bbox": [0, 0, 10, 10], "confidence": 0.9, "inis_durumu": 0, "engel_var": True},
                ],
            },
            "position_estimation": None,
            "matched_objects": [],
            "processing_time_ms": 5.0,
        }

        with unittest.mock.patch.object(self.comm, 'send_results', return_value=True) as mock:
            self.comm.send_frame_results(result)
            dets = mock.call_args[0][1]
            for d in dets:
                if d["class_id"] == 3:
                    self.assertEqual(d["landing_status"], 0)


if __name__ == "__main__":
    unittest.main()
