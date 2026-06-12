#!/usr/bin/env python3
"""One-shot demo: train → diagnose sample images → show ledger."""

from pathlib import Path

from blockchain import PoultryLedger
from config import DATA_DIR
from inference import DiseaseDetector
from train import generate_synthetic_dataset, train


def main() -> None:
    print("=" * 60)
    print("Decentralized Poultry Disease Detection — Demo")
    print("=" * 60)

    if not (DATA_DIR / "healthy").exists():
        print("\n[1/4] Generating synthetic flock images...")
        generate_synthetic_dataset(DATA_DIR, per_class=80)
    else:
        print("\n[1/4] Using existing dataset")

    model_path = Path("models/poultry_cnn.keras")
    if not model_path.exists():
        print("[2/4] Training CNN (augmentation + class weights)...")
        train(DATA_DIR, epochs=8)
    else:
        print("[2/4] Using existing trained model")

    ledger = PoultryLedger()
    detector = DiseaseDetector(ledger=ledger)

    print("[3/4] Running inference on sample birds...")
    samples = [
        ("farm_alpha", str(next((DATA_DIR / "healthy").glob("*.png")))),
        ("farm_beta", str(next((DATA_DIR / "diseased").glob("*.png")))),
    ]

    for farm_id, img_path in samples:
        result = detector.diagnose(img_path, farm_id=farm_id)
        print(f"\n  Farm: {farm_id}")
        print(f"  Image: {Path(img_path).name}")
        print(f"  → {result['label']} ({result['confidence']:.1%})")
        if result["alert"]:
            print(f"  ⚠ ALERT: {result['alert']}")
        print(f"  Ledger block #{result['ledger_block']} hash={result['ledger_hash']}")

    print(f"\n[4/4] Ledger integrity: {'VALID' if ledger.verify() else 'BROKEN'}")
    print(f"      Total blocks: {ledger.length}")
    print("\nStart API: uvicorn api:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
