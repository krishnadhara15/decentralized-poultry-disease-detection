"""OpenCV preprocessing and augmentation for field poultry images."""

import cv2
import numpy as np

from config import IMAGE_SIZE


def load_image(path: str) -> np.ndarray:
    """Load BGR image from disk; raise if unreadable."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_and_normalize(img: np.ndarray) -> np.ndarray:
    """Resize to model input size and scale pixels to [0, 1]."""
    resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_for_inference(path: str) -> np.ndarray:
    """Full inference pipeline: load → resize → normalize → batch dim."""
    img = load_image(path)
    tensor = resize_and_normalize(img)
    return np.expand_dims(tensor, axis=0)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Field-safe augmentations to combat rare diseased examples:
    horizontal flip, small rotation, brightness jitter.
    """
    out = img.copy()

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    angle = rng.uniform(-15, 15)
    h, w = out.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    factor = rng.uniform(0.8, 1.2)
    out = np.clip(out * factor, 0, 255).astype(np.uint8)

    return out


def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply augmentation to a batch of uint8 BGR images."""
    return np.stack([augment(img, rng) for img in images])
