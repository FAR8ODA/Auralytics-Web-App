# 🔊 Auralytics

**Acoustic anomaly detection for predictive maintenance.**

Auralytics listens to industrial machine audio and flags abnormal behavior before failure occurs. A convolutional autoencoder is trained on normal machine sounds — when a new clip reconstructs poorly, that reconstruction error is the anomaly signal. The project ships with a live web demo: upload a `.wav` clip, pick a machine type, and get back an anomaly score and a spectrogram visualization.

---

## How It Works

```
raw audio (.wav)
      │
      ▼
log-mel spectrogram  (128 mels · FFT 1024 · hop 512 · 16 kHz)
      │
      ▼
convolutional autoencoder  (trained on normal clips only)
      │
      ▼
reconstruction error (MSE)
      │
      ▼
anomaly score  ──►  NORMAL / ANOMALOUS verdict
```

High reconstruction error means the model has never seen sounds like this — likely anomalous.

---

## Dataset

[DCASE 2020 Task 2](https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds) — Unsupervised Anomalous Sound Detection in Machine Condition Monitoring.

| Machine | Train (normal) | Test normal | Test anomalous |
|---------|---------------|-------------|----------------|
| Fan     | 3 675         | 400         | 1 475          |
| Pump    | 3 349         | 400         | 456            |
| Valve   | 3 291         | 400         | 479            |

Download: [zenodo.org/records/3678171](https://zenodo.org/records/3678171) — see [`data/README.md`](data/README.md) for setup instructions.

---

## Project Structure

```
auralytics/
├── src/
│   ├── preprocess.py   wav → log-mel spectrogram pipeline
│   ├── dataset.py      PyTorch Dataset + DataLoader factory
│   ├── model.py        convolutional autoencoder
│   ├── train.py        training loop with early stopping + checkpointing
│   ├── evaluate.py     AUC-ROC, pAUC, F1 evaluation
│   └── utils.py        plotting, seeding, checkpointing
│
├── notebooks/
│   ├── 01_eda.ipynb          dataset inspection & spectrogram visualization
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb     Colab-ready training walkthrough
│   └── 04_evaluation.ipynb   AUC curves, score distributions, failure cases
│
├── backend/
│   ├── main.py         FastAPI — POST /predict, GET /health
│   └── inference.py    model load + audio → score pipeline
│
├── frontend/
│   ├── index.html      upload UI + score + spectrogram display
│   ├── style.css
│   └── app.js          calls backend, renders results
│
├── data/               raw + processed audio (gitignored — download separately)
├── models/             trained .pth checkpoints (gitignored)
└── results/            figures, logs, evaluation outputs
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

See [`data/README.md`](data/README.md) for the download links and expected folder layout.

### 3. Preprocess audio

```bash
python src/preprocess.py --machine_types fan pump valve
```

Converts raw `.wav` files into normalized log-mel spectrograms (`.npy`) under `data/processed/`.

### 4. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Train (recommended: Google Colab free GPU)

```bash
python -m src.train --machine_type fan --epochs 50
```

Saves the best checkpoint to `models/fan_best.pth`.

### 6. Evaluate

```bash
python -m src.evaluate --machine_type fan
```

### 7. Run the web demo

```bash
# Terminal 1 — backend
cd backend && uvicorn main:app --reload

# Terminal 2 — open the frontend
open frontend/index.html   # or just double-click it
```

---

## Evaluation

Following the official DCASE 2020 Task 2 protocol:

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Primary — threshold-free discrimination between normal and anomalous |
| **pAUC** | Partial AUC over FPR 0–0.1, penalizes false alarms in safety-critical settings |
| **F1** | Reported at the threshold that maximizes it |

Target: AUC > 0.75 across all three machine types.

---

## Tech Stack

PyTorch · librosa · FastAPI · Vanilla JS · DCASE 2020

---

*CPEN 355 Final Project — UBC Electrical and Computer Engineering*
