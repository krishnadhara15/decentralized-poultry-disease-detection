"""CNN classifier: healthy vs diseased poultry (whole-image classification)."""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import INPUT_SHAPE, NUM_CLASSES


def build_cnn() -> tf.keras.Model:
    """
    Lightweight CNN — ~5 conv blocks worth of depth for interview walkthrough:
    Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool → Flatten → Dense(128) → Softmax(2)
    Input: 224×224×3
    """
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            # Block 1 — low-level edges / texture
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Block 2 — plumage patterns
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Block 3 — posture / lesion cues
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="poultry_disease_cnn",
    )
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile with class-weight-friendly sparse categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
