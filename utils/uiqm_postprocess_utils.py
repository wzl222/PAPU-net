from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(name):
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def apply_clahe(image, clip_limit, tile_grid_size=8):
    if clip_limit <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size)
    )
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def adjust_saturation(image, saturation):
    if abs(saturation - 1.0) < 1e-6:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_contrast(image, contrast):
    if abs(contrast - 1.0) < 1e-6:
        return image
    image_float = image.astype(np.float32)
    mean = image_float.mean(axis=(0, 1), keepdims=True)
    return np.clip((image_float - mean) * contrast + mean, 0, 255).astype(np.uint8)


def unsharp_mask(image, amount, sigma=1.0):
    if amount <= 0:
        return image
    image_float = image.astype(np.float32)
    blur = cv2.GaussianBlur(image_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = image_float + amount * (image_float - blur)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_rgb_uint8(image, clahe, saturation, contrast, sharpness):
    enhanced = apply_clahe(image, clahe)
    enhanced = adjust_saturation(enhanced, saturation)
    enhanced = adjust_contrast(enhanced, contrast)
    enhanced = unsharp_mask(enhanced, sharpness)
    return enhanced


def enhance_rgb_float(image, clahe, saturation, contrast, sharpness):
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    enhanced = enhance_rgb_uint8(image_u8, clahe, saturation, contrast, sharpness)
    return enhanced.astype(np.float32) / 255.0


def preset_params(name):
    presets = {
        "mild": {
            "clahe": 0.0,
            "saturation": 1.0,
            "contrast": 1.0,
            "sharpness": 0.45,
        },
        "strong": {
            "clahe": 1.2,
            "saturation": 1.0,
            "contrast": 1.0,
            "sharpness": 0.45,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}")
    return presets[name].copy()
