import cv2
import numpy as np


def is_night_image(img, brightness_threshold=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return mean_brightness < brightness_threshold


def is_thermal_colormap(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()
    val_mean = hsv[:, :, 2].mean()
    gray_std = gray.std()
    is_colormap = sat_mean > 80 and val_mean > 100 and gray_std < 60
    return is_colormap


def preprocess_frame(img):
    processed = img.copy()
    height, width = processed.shape[:2]

    if is_night_image(processed):
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gamma = 1.5
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    if is_thermal_colormap(processed):
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)
        gamma = 1.2
        normalized = np.power(normalized / 255.0, 1.0 / gamma)
        normalized = (normalized * 255).astype(np.uint8)
        processed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    return processed
