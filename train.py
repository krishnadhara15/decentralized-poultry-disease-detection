"""Train CNN on poultry images with augmentation and class weighting."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    RANDOM_SEED,
)
from model import build_cnn, compile_model
from preprocessing import augment


def _collect_images(data_dir: Path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = data_dir / cls
        if not folder.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in folder.glob(ext):
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def _load_batch(paths: list[str], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    images, ys = [], []
    for path, label in zip(paths, labels):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)
        ys.append(label)
    return np.array(images, dtype=np.float32) / 255.0, np.array(ys)


def generate_synthetic_dataset(out_dir: Path, per_class: int = 80) -> None:
    """
    Bootstrap demo data when no real flock photos are available.
    Healthy = green-tinted blobs; diseased = red spots + dull tone.
    Replace with labeled field photos for production (100k+ birds coverage).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = out_dir / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)

            if cls == "healthy":
                # Bright pasture + uniform plumage — clear healthy cues
                img[:] = (55, 145, 75)
                cx, cy = rng.integers(70, 154, size=2)
                axes = (rng.integers(45, 65), rng.integers(35, 50))
                cv2.ellipse(img, (int(cx), int(cy)), axes, rng.integers(0, 360), 0, 360, (70, 185, 90), -1)
                cv2.ellipse(img, (int(cx - 25), int(cy - 20)), (12, 10), 0, 0, 360, (220, 220, 100), -1)
            else:
                # Dull tone + red lesions — diseased visual markers
                img[:] = (45, 55, 65)
                cx, cy = rng.integers(70, 154, size=2)
                axes = (rng.integers(45, 65), rng.integers(35, 50))
                cv2.ellipse(img, (int(cx), int(cy)), axes, rng.integers(0, 360), 0, 360, (55, 60, 70), -1)
                for _ in range(rng.integers(5, 12)):
                    px, py = rng.integers(55, 170, size=2)
                    cv2.circle(img, (int(px), int(py)), rng.integers(6, 16), (30, 30, 220), -1)
                    cv2.circle(img, (int(px), int(py)), rng.integers(2, 5), (0, 0, 255), -1)

            noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(folder / f"{cls}_{i:03d}.png"), img)


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Up-weight rare diseased class — key for imbalanced flock data."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = counts.sum()
    return {i: float(total / (len(CLASS_NAMES) * c)) if c else 1.0 for i, c in enumerate(counts)}


def train(data_dir: Path, epochs: int = EPOCHS) -> Path:
    paths, labels = _collect_images(data_dir)
    if len(paths) < 10:
        raise RuntimeError(f"Need images under {data_dir}/healthy and .../diseased")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(paths))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Load uint8 for augmentation, then normalize
    train_imgs_u8 = []
    for p in train_paths:
        img = cv2.imread(p)
        train_imgs_u8.append(cv2.resize(img, IMAGE_SIZE))
    train_imgs_u8 = np.array(train_imgs_u8)

    aug_rng = np.random.default_rng(RANDOM_SEED)
    augmented = []
    aug_labels = []
    for img, lbl in zip(train_imgs_u8, train_labels):
        augmented.append(img)
        aug_labels.append(lbl)
        # 2× oversample diseased via augmentation
        copies = 2 if lbl == CLASS_NAMES.index("diseased") else 1
        for _ in range(copies):
            augmented.append(augment(img, aug_rng))
            aug_labels.append(lbl)

    x_train = np.array(augmented, dtype=np.float32) / 255.0
    y_train = np.array(aug_labels)

    x_val, y_val = _load_batch(val_paths, val_labels)

    class_weights = compute_class_weights(train_labels)

    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1,
    )

    cnn = compile_model(build_cnn(), learning_rate=LEARNING_RATE)

    cnn.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED),
        validation_data=(x_val, y_val),
        epochs=epochs,
        class_weight=class_weights,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "poultry_cnn.keras"
    cnn.save(out)
    print(f"Saved model → {out}")
    print(f"Class weights used: {class_weights}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train poultry disease CNN")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--synthetic", action="store_true", help="Generate demo dataset")
    args = parser.parse_args()

    if args.synthetic or not (args.data / "healthy").exists():
        print("Generating synthetic demo dataset...")
        generate_synthetic_dataset(args.data)

    train(args.data, epochs=args.epochs)


if __name__ == "__main__":
    main()
