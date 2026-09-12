import numpy as np
import cv2


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_mAP(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    total_ground_truths = len(ground_truths)

    matched = [False] * len(ground_truths)
    for pred in predictions:
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(ground_truths):
            if not matched[j] and pred["class_id"] == gt["class_id"]:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

        if best_iou >= iou_threshold and best_idx >= 0:
            true_positives += 1
            matched[best_idx] = True
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0.0

    return {"precision": precision, "recall": recall, "mAP@0.5": (precision + recall) / 2 if (precision + recall) > 0 else 0.0}


def compute_position_error(estimated, ground_truth):
    dx = estimated[0] - ground_truth[0]
    dy = estimated[1] - ground_truth[1]
    dz = estimated[2] - ground_truth[2]
    return {"dx": dx, "dy": dy, "dz": dz, "drift_xy": np.sqrt(dx ** 2 + dy ** 2), "drift_z": abs(dz)}


def compute_matching_accuracy(total_attempts, successful_matches):
    if total_attempts == 0:
        return 0.0
    return successful_matches / total_attempts * 100


def compute_fps(total_frames, total_time_seconds):
    if total_time_seconds <= 0:
        return 0.0
    return total_frames / total_time_seconds
