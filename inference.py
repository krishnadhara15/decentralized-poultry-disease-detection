"""End-to-end inference: preprocess → CNN → alert → ledger."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from alerts import evaluate_and_alert
from blockchain import HealthRecord, PoultryLedger
from config import CLASS_NAMES, MODEL_DIR
from preprocessing import preprocess_for_inference


class DiseaseDetector:
    def __init__(self, model_path: Path | None = None, ledger: PoultryLedger | None = None):
        path = model_path or MODEL_DIR / "poultry_cnn.keras"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python train.py"
            )
        self.model = tf.keras.models.load_model(path)
        self.ledger = ledger or PoultryLedger()

    def predict(self, image_path: str) -> tuple[str, float, list[float]]:
        batch = preprocess_for_inference(image_path)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

    def diagnose(
        self,
        image_path: str,
        farm_id: str,
        channel: str = "dashboard",
    ) -> dict:
        label, confidence, probs = self.predict(image_path)
        alert = evaluate_and_alert(farm_id, label, confidence, probs, channel=channel)

        record = HealthRecord(
            farm_id=farm_id,
            image_path=image_path,
            label=label,
            confidence=confidence,
            alert_triggered=alert is not None,
            timestamp=time.time(),
            metadata={"probabilities": dict(zip(CLASS_NAMES, probs))},
        )
        block = self.ledger.append(record)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": dict(zip(CLASS_NAMES, [round(p, 4) for p in probs])),
            "alert": alert.message if alert else None,
            "ledger_block": block.index,
            "ledger_hash": block.hash[:16] + "...",
        }
