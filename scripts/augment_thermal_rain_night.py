import cv2
import numpy as np
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "data", "custom_uap_uai")
DST_DIR = os.path.join(BASE_DIR, "data", "augmented_conditions")

random.seed(42)
np.random.seed(42)


def add_thermal_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def add_rain_effect(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    num_drops = random.randint(80, 200)
    xs = np.random.randint(0, w - 1, num_drops)
    ys = np.random.randint(0, h - 1, num_drops)
    lengths = np.random.randint(10, 25, num_drops)
    for i in range(num_drops):
        x, y, l = xs[i], ys[i], lengths[i]
        cv2.line(mask, (x, y), (x - 1, y + l), 255, 1)
    rain = np.stack([mask // 2, mask // 2, mask], axis=2).astype(np.uint8)
    result = cv2.addWeighted(img, 1.0, rain, 0.3, 0)
    result = cv2.GaussianBlur(result, (3, 3), 0.3)
    return result


def add_night_effect(img):
    result = cv2.convertScaleAbs(img, alpha=0.25, beta=random.randint(5, 20))
    noise = np.random.normal(0, 15, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - 0.4 * (dist / max_dist)
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] * vignette, 0, 255).astype(np.uint8)
    return result


AUGMENTATIONS = {
    "thermal": add_thermal_effect,
    "rain": add_rain_effect,
    "night": add_night_effect,
}


def main():
    for cond in AUGMENTATIONS:
        os.makedirs(os.path.join(DST_DIR, cond, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST_DIR, cond, "labels"), exist_ok=True)

    images = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith((".png", ".jpg"))])
    print(f"Found {len(images)} source images")

    generated = 0
    for img_name in images:
        stem, ext = os.path.splitext(img_name)
        is_uap = "uap" in stem.lower()
        class_id = 2 if is_uap else 3

        label_name = stem + ".txt"
        src_label = os.path.join(SRC_DIR, "labels", label_name)
        label_content = f"{class_id} 0.5 0.5 0.6 0.6\n"
        if os.path.exists(src_label):
            with open(src_label, "r") as f:
                label_content = f.read()

        img_path = os.path.join(SRC_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        for aug_name, aug_fn in AUGMENTATIONS.items():
            aug_img = aug_fn(img)
            out_name = f"{stem}_{aug_name}{ext}"
            cv2.imwrite(os.path.join(DST_DIR, aug_name, "images", out_name), aug_img)
            with open(os.path.join(DST_DIR, aug_name, "labels", out_name.replace(ext, ".txt")), "w") as f:
                f.write(label_content)
            generated += 1

    print(f"Generated {generated} images across {len(AUGMENTATIONS)} conditions.")
    for cond in AUGMENTATIONS:
        n = len(os.listdir(os.path.join(DST_DIR, cond, "images")))
        print(f"  {cond}: {n} images")


if __name__ == "__main__":
    main()
