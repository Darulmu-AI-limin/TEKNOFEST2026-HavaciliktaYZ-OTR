from ..utils.metrics import calculate_iou


def analyze_landing(detections, landing_zones):
    uap_detected = []
    uai_detected = []
    tasit_detected = []

    for det in detections:
        if det["class_id"] == 2:
            uap_detected.append(det)
        elif det["class_id"] == 3:
            uai_detected.append(det)
        elif det["class_id"] == 0:
            tasit_detected.append(det)

    landing_analysis = {
        "uap_status": [],
        "uai_status": [],
        "tasit_on_landing": [],
    }

    for landing_type, detected, results_key in [
        ("UAP", uap_detected, "uap_status"),
        ("UAI", uai_detected, "uai_status"),
    ]:
        for det in detected:
            det_box = det["bbox"]
            is_blocked = False
            blocking_object = None

            for zone in landing_zones:
                zone_box = zone["bbox"]
                overlap = calculate_iou(det_box, zone_box)
                if overlap > 0.05:
                    for tasit in tasit_detected:
                        tasit_box = tasit["bbox"]
                        tasit_overlap = calculate_iou(tasit_box, zone_box)
                        if tasit_overlap > 0.05:
                            is_blocked = True
                            blocking_object = {"type": "tasit", "confidence": tasit["confidence"]}
                            break

            landing_analysis[results_key].append({
                "bbox": det_box,
                "confidence": det["confidence"],
                "inis_durumu": 0 if is_blocked else 1,
                "engel_var": is_blocked,
                "engel": blocking_object,
            })

    return landing_analysis


def check_gps_health(gps_data):
    if gps_data is None:
        return 0
    return gps_data.get("health_status", 0)
