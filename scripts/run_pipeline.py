import sys
import os
import json
import time
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.train_yolo import load_model, detect
from src.integration.pipeline import BlackScopePipeline
from src.integration.server_comm import ServerCommunicator
from src.utils.visualization import draw_detections, draw_landing_status, draw_optical_flow


def load_camera_params(path="configs/server_config.json"):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    fx = cfg.get("camera_fx", 458.654)
    fy = cfg.get("camera_fy", 457.296)
    cx = cfg.get("camera_cx", 367.215)
    cy = cfg.get("camera_cy", 248.375)

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def warmup_model(model, n=3):
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    for i in range(n):
        _ = detect(model, img)
    print(f"[OK] Model warmup: {n} frames")


def competition_mode(pipeline, communicator, server_url, max_frames=2250):
    print(f"[Competition] Connected to {server_url}")
    print(f"[Competition] Waiting for frames...")

    frame_count = 0
    consecutive_failures = 0

    while frame_count < max_frames:
        frame_data = communicator.get_frame()
        if frame_data is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print("[!] Too many failures, exiting")
                break
            time.sleep(0.5)
            continue

        consecutive_failures = 0
        frame_url = frame_data.get("url", "")
        image_url = frame_data.get("image_url", "")
        video_name = frame_data.get("video_name", "")
        session = frame_data.get("session", "")

        server_frame_id = frame_data.get("frame_id", frame_data.get("url", frame_count))
        translation_x = frame_data.get("translation_x", 0.0)
        translation_y = frame_data.get("translation_y", 0.0)
        translation_z = frame_data.get("translation_z", 0.0)
        gps_health = frame_data.get("gps_health_status", 1)

        img_bytes = communicator.fetch_image(image_url)
        if img_bytes is None:
            print(f"[!] Failed to fetch image: {image_url}")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = dummy
        else:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[!] Failed to decode image: {image_url}")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        gps_data = {
            "health_status": gps_health,
            "altitude": abs(translation_z) if translation_z != 0 else 10.0,
        }
        landing_zones = []

        reference_position = None
        if gps_health == 1:
            reference_position = {
                "x": translation_x,
                "y": translation_y,
                "z": translation_z,
            }

        result = pipeline.process_frame(frame, gps_data, landing_zones, reference_position)

        detections = result["detections"]
        pos = result.get("position_estimation", None)
        translations = None
        if pos:
            p = pos.get("position", [0, 0, 0])
            translations = {"x": p[0], "y": p[1], "z": p[2]}
        matched = result.get("matched_objects", [])
        undefined_objects = [{"id": m.get("id", ""), "bbox": m.get("bbox", [0, 0, 0, 0])} for m in matched]

        success = communicator.send_results(
            frame_id=server_frame_id,
            detections=detections,
            translations=translations,
            undefined_objects=undefined_objects,
            user_url=session,
            frame_url=frame_url,
        )

        frame_count += 1
        avg_fps = pipeline.get_average_fps()
        status = "OK" if success else "FAIL"
        print(f"[{frame_count}/{max_frames}] {status} | det={len(detections)} vo={pos is not None} match={len(matched)} fps={avg_fps:.1f}", flush=True)

    print(f"[Competition] Done. Processed {frame_count} frames")


def video_mode(pipeline, communicator, video_source, server_url, simulate_gps_failure=False):
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        source_name = f"Camera {video_source}"
    else:
        cap = cv2.VideoCapture(video_source)
        source_name = video_source

    if not cap.isOpened():
        print(f"[!] Cannot open source: {source_name}")
        sys.exit(1)

    print(f"[VideoMode] Playing: {source_name}")
    print(f"[VideoMode] Server: {server_url}")
    print("[VideoMode] Press 'q' to quit, 's' to save frame")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "results", "detection")
    os.makedirs(output_dir, exist_ok=True)
    save_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] End of video stream")
                break

            gps_health = 0 if simulate_gps_failure else 1
            gps_data = {"health_status": gps_health, "altitude": 10.0}

            landing_zones = [{"bbox": [100, 100, 300, 300], "type": "UAP"}]

            result = pipeline.process_frame(frame, gps_data, landing_zones)

            vis = frame.copy()
            if result["detections"]:
                vis = draw_detections(vis, result["detections"])

            if result.get("landing_analysis"):
                for uap in result["landing_analysis"].get("uap_status", []):
                    vis = draw_landing_status(vis, uap["inis_durumu"], uap["bbox"])

            avg_fps = pipeline.get_average_fps()
            cv2.putText(vis, f"FPS: {avg_fps:.2f} | Frame: {result['frame_id']}", (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("BlackScope - TEKNOFEST 2026", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[VideoMode] Shutting down...")
                break
            elif key == ord('s'):
                save_path = os.path.join(output_dir, f"frame_{result['frame_id']:06d}.jpg")
                cv2.imwrite(save_path, vis)
                print(f"[VideoMode] Saved: {save_path}")
                save_counter += 1

            communicator.send_frame_results(result)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        communicator.close()

    print(f"[VideoMode] Processed {result['frame_id']} frames")
    print(f"[VideoMode] Average FPS: {pipeline.get_average_fps():.2f}")


def main():
    parser = argparse.ArgumentParser(description="BlackScope TEKNOFEST 2026 Pipeline")
    parser.add_argument("--mode", choices=["competition", "video"], default="video",
                        help="'competition' = GET/POST loop, 'video' = local file/camera (default: video)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file for video mode")
    parser.add_argument("--simulate-gps-failure", action="store_true",
                        help="Simulate GPS failure (health_status=0) to test visual odometry")
    parser.add_argument("--max-frames", type=int, default=2250,
                        help="Max frames in competition mode (default: 2250)")
    parser.add_argument("--server", type=str, default=None,
                        help="Server URL (overrides SERVER_URL env)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "yolo11m_visdrone.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "yolo11m.pt")
        if not os.path.exists(model_path):
            print(f"[!] Model not found at {model_path}")
            sys.exit(1)

    print(f"[BlackScope] Loading model: {model_path}")
    model = load_model(model_path)

    try:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
            print(f"[OK] Model moved to CUDA: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    warmup_model(model)

    camera_matrix = load_camera_params()
    pipeline = BlackScopePipeline(
        detection_model=model,
        camera_matrix=camera_matrix,
        reference_objects=[],
    )

    server_url = args.server or os.environ.get("SERVER_URL", "http://127.0.0.25:5000")
    communicator = ServerCommunicator(server_url)

    if args.mode == "competition":
        competition_mode(pipeline, communicator, server_url, max_frames=args.max_frames)
    else:
        video_source = args.video or os.environ.get("VIDEO_SOURCE", "0")
        video_mode(pipeline, communicator, video_source, server_url, args.simulate_gps_failure)


if __name__ == "__main__":
    main()
