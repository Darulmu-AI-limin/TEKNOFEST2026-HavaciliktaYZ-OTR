from ultralytics import YOLO
import os
import sys
import json


def load_visdrone_yaml():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base, "configs", "visdrone.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    alt = os.path.join(base, "data", "visdrone", "VisDrone.yaml")
    if os.path.exists(alt):
        return alt
    return None


def train():
    print("=" * 60)
    print("BlackScope - YOLO11m Eg timi (VisDrone2019 + Ozel Veri Seti)")
    print("=" * 60)

    visdrone_yaml = load_visdrone_yaml()
    if visdrone_yaml is None:
        print("[!] VisDrone.yaml not found. Starting with COCO pretrained model.")
        print("[!] Download VisDrone first: python scripts/download_data.py")
        data_config = None
    else:
        data_config = visdrone_yaml
        print(f"[+] Using data config: {data_config}")

    model = YOLO("yolo11m.pt")

    params = {
        "data": data_config,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 20,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "project": "blackscope_detection",
        "name": "yolo11m_visdrone",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "SGD",
        "cos_lr": True,
        "device": 0,
        "workers": 4,
    }

    if data_config is None:
        params.pop("data", None)

    print("\nTraining parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print("\n[+] Training started...")
    results = model.train(**params)

    print("\n[+] Training complete!")
    print(f"[+] Best model saved to: {results.save_dir}/weights/best.pt")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    best_path = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, os.path.join(model_dir, "best.pt"))
        print(f"[+] Model copied to: {os.path.join(model_dir, 'best.pt')}")

    print("\n[+] Validation results:")
    print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


if __name__ == "__main__":
    train()
