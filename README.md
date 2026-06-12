# Decentralized Poultry Disease Detection

**Computer Vision · Deep Learning · Blockchain** · Mar 2024

CNN-based disease detection for 100,000+ birds with blockchain-backed tamper-proof health records and real-time alerting.

**Stack:** CNN (TensorFlow/Keras) · OpenCV · FastAPI · Append-only blockchain ledger

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --synthetic --epochs 12
uvicorn api:app --host 127.0.0.1 --port 8001
```

Open **http://127.0.0.1:8001** for the styled dashboard (upload images, view ledger, alerts).

Or run the CLI demo:

```bash
python demo.py
```

Train on your own flock photos:

```
data/
  healthy/   ← field/camera images
  diseased/
```

```bash
python train.py --data data --epochs 10
uvicorn api:app --reload
```

## Architecture

```
Camera/field photo
       ↓
OpenCV preprocess (resize 224×224, normalize, augment)
       ↓
CNN classifier → healthy | diseased + confidence
       ↓
Threshold → alert (dashboard/SMS hook)
       ↓
SHA-256 hash chain → shared ledger (all participating farms)
```

### CNN (whole-image classification)

| Layer | Output size |
|-------|-------------|
| Input | 224×224×3 |
| Conv2D(32) + ReLU | 224×224×32 |
| MaxPool | 112×112×32 |
| Conv2D(64) + ReLU | 112×112×64 |
| MaxPool | 56×56×64 |
| Conv2D(128) + ReLU | 56×56×128 |
| MaxPool | 28×28×128 |
| Flatten → Dense(128) → Softmax(2) | 2 classes |

**Not object detection** — one label per image. Detection = bounding boxes; only claim it if you built it.

### Rare diseased examples

- OpenCV augmentation (flip, rotation, brightness)
- Keras `ImageDataGenerator` on training batches
- Extra augmented copies for the diseased class
- `class_weight` in `fit()` to up-weight minority class
- Confidence threshold tuning (`ALERT_THRESHOLD = 0.70`)

### Why blockchain?

**Scoped answer:** Multiple independent farms needed health records no single party controlled. Hash-linked blocks detect tampering across participants. **One owner? Use a normal database.**

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model + ledger status |
| `/diagnose` | POST | Upload image + `farm_id` → diagnosis + ledger entry |
| `/ledger` | GET | Full chain |
| `/ledger/farm/{id}` | GET | Records for one farm |

## Interview cheat sheet

- **100,000+ birds** = flock coverage across monitored farms, not training set size
- **~50% outbreak reduction** = before/after deployment comparison (define who measured and the period)
- **Alerting:** confidence ≥ threshold → event → notification channel
- **Classification vs detection:** this project is classification

## Project layout

```
config.py          # thresholds, paths, input size
preprocessing.py   # OpenCV load/resize/augment
model.py           # CNN architecture
train.py           # training + synthetic demo data
inference.py       # predict → alert → ledger
blockchain.py      # append-only hash chain
alerts.py          # threshold-based notifications
api.py             # FastAPI + dashboard
static/            # index.html, css/style.css, js/app.js
demo.py            # end-to-end demo
```
