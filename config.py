"""Project configuration — tuned for field poultry imaging."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LEDGER_DIR = ROOT / "ledger"

# CNN input: 224x224 RGB (standard for transfer-learning-style CNNs)
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

# Class imbalance: diseased samples are rare in the field
CLASS_NAMES = ["healthy", "diseased"]
NUM_CLASSES = len(CLASS_NAMES)

# Alert when CNN predicts diseased with confidence above this threshold
ALERT_THRESHOLD = 0.70

# Augmentation seed for reproducibility
RANDOM_SEED = 42

# Training defaults (fast baseline — swap in real flock data for production)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
