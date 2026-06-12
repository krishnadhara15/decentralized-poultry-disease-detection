"""FastAPI service — dashboard + diagnosis + shared ledger."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from blockchain import PoultryLedger
from config import ALERT_THRESHOLD, ROOT
from inference import DiseaseDetector

PROJECT_TITLE = "Decentralized Poultry Disease Detection"
PROJECT_TAGS = "Computer Vision · Deep Learning · Blockchain"

app = FastAPI(
    title=PROJECT_TITLE,
    description=(
        f"{PROJECT_TAGS} — CNN classification with blockchain-backed "
        "flock health records and real-time alerting (Mar 2024)."
    ),
    version="1.0.0",
)

STATIC_DIR = ROOT / "static"
ledger = PoultryLedger()
detector: DiseaseDetector | None = None

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def load_model() -> None:
    global detector
    try:
        detector = DiseaseDetector(ledger=ledger)
    except FileNotFoundError:
        detector = None


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": PROJECT_TITLE,
        "tags": PROJECT_TAGS,
        "model_loaded": detector is not None,
        "ledger_blocks": ledger.length,
        "ledger_valid": ledger.verify(),
        "alert_threshold": ALERT_THRESHOLD,
    }


@app.post("/diagnose")
async def diagnose(
    farm_id: str = Form(...),
    image: UploadFile = File(...),
    channel: str = Form(default="dashboard"),
):
    if detector is None:
        raise HTTPException(503, "Model not trained. Run: python train.py --synthetic")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    result = detector.diagnose(tmp_path, farm_id=farm_id, channel=channel)
    Path(tmp_path).unlink(missing_ok=True)
    return result


@app.get("/ledger")
def get_ledger():
    return {
        "length": ledger.length,
        "valid": ledger.verify(),
        "blocks": [
            {
                "index": b.index,
                "farm_id": b.record.farm_id,
                "label": b.record.label,
                "confidence": b.record.confidence,
                "alert": b.record.alert_triggered,
                "hash": b.hash,
            }
            for b in ledger.chain
        ],
    }


@app.get("/ledger/farm/{farm_id}")
def farm_records(farm_id: str):
    records = ledger.records_for_farm(farm_id)
    return {
        "farm_id": farm_id,
        "count": len(records),
        "records": [
            {
                "farm_id": r.farm_id,
                "label": r.label,
                "confidence": r.confidence,
                "alert_triggered": r.alert_triggered,
                "timestamp": r.timestamp,
            }
            for r in records
        ],
    }
