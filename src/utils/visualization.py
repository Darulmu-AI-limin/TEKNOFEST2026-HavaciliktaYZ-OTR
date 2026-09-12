import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0)
]


CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI"
}


def draw_detections(frame, detections, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        class_id = det["class_id"]
        conf = det.get("confidence", 1.0)
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(class_id, class_id)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def draw_landing_status(frame, status, landing_zone):
    vis = frame.copy()
    if landing_zone:
        x1, y1, x2, y2 = map(int, landing_zone)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "Inis Alani", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if status == 0:
        cv2.putText(vis, "INIS UYGUN DEGIL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif status == 1:
        cv2.putText(vis, "INIS UYGUN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def draw_optical_flow(frame, prev_points, next_points, status):
    vis = frame.copy()
    if prev_points is None or next_points is None:
        return vis
    for i, (prev, nxt) in enumerate(zip(prev_points, next_points)):
        if status[i]:
            a, b = prev.ravel()
            c, d = nxt.ravel()
            a, b, c, d = map(int, [a, b, c, d])
            cv2.line(vis, (a, b), (c, d), (0, 255, 0), 1)
            cv2.circle(vis, (c, d), 2, (0, 0, 255), -1)
    return vis


def draw_matches(img1, img2, kp1, kp2, matches, max_draw=50):
    return cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


def plot_results(results_dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if "mAP" in results_dict:
        axes[0, 0].bar(["mAP@0.5", "Precision", "Recall"], [results_dict.get("mAP@0.5", 0), results_dict.get("precision", 0), results_dict.get("recall", 0)])
        axes[0, 0].set_title("Detection Performance")
        axes[0, 0].set_ylim([0, 1])
    if "drift_xy" in results_dict:
        axes[0, 1].bar(["X/Y Drift", "Z Drift"], [results_dict.get("drift_xy", 0), results_dict.get("drift_z", 0)])
        axes[0, 1].set_title("Position Estimation Error (m)")
    if "matching_accuracy" in results_dict:
        axes[1, 0].bar(["Accuracy"], [results_dict.get("matching_accuracy", 0)])
        axes[1, 0].set_title("Matching Accuracy (%)")
        axes[1, 0].set_ylim([0, 100])
    if "fps" in results_dict:
        axes[1, 1].bar(["FPS"], [results_dict.get("fps", 0)])
        axes[1, 1].set_title("Processing Speed (FPS)")
        axes[1, 1].set_ylim([0, max(2, results_dict.get("fps", 0) * 1.5)])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_result_image(frame, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
