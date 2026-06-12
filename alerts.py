"""Confidence-threshold alerting for farm operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import ALERT_THRESHOLD, CLASS_NAMES


@dataclass
class Alert:
    farm_id: str
    label: str
    confidence: float
    message: str
    channel: str


# In production: plug in SMS (Twilio), push, or dashboard webhook
_alert_handlers: list[Callable[[Alert], None]] = []


def register_handler(handler: Callable[[Alert], None]) -> None:
    _alert_handlers.append(handler)


def default_console_handler(alert: Alert) -> None:
    print(f"[ALERT][{alert.channel}] farm={alert.farm_id} | {alert.message}")


register_handler(default_console_handler)


def evaluate_and_alert(
    farm_id: str,
    label: str,
    confidence: float,
    probabilities: list[float],
    threshold: float = ALERT_THRESHOLD,
    channel: str = "dashboard",
) -> Alert | None:
    """
    Alert when CNN predicts diseased AND predicted-class confidence ≥ threshold.
    """
    if label != "diseased" or confidence < threshold:
        return None

    diseased_idx = CLASS_NAMES.index("diseased")
    diseased_prob = probabilities[diseased_idx]

    alert = Alert(
        farm_id=farm_id,
        label="diseased",
        confidence=round(confidence, 4),
        message=(
            f"Diseased poultry detected (confidence {confidence:.1%}). "
            "Inspect flock immediately and isolate affected birds."
        ),
        channel=channel,
    )
    for handler in _alert_handlers:
        handler(alert)
    return alert
