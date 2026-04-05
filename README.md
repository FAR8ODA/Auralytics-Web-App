# 🔊 Auralytics

**Acoustic anomaly detection for predictive maintenance.**

Auralytics listens to industrial machine audio and flags abnormal behavior before failure occurs. Built on the DCASE 2020 Task 2 benchmark, it trains a convolutional autoencoder on normal machine sounds — when a new clip reconstructs poorly, that reconstruction error is the anomaly signal.

The project ships with a live web demo: upload a `.wav` clip, pick a machine type, and get back an anomaly score, a verdict, and a spectrogram visualization.

---

## Demo

> Upload a `.wav` → get an anomaly score + spectrogram explanation

*(Demo link coming after model training — see roadmap)*

---

## How It Works

```
raw audio (.wav)
      │
      ▼
log-mel spectrogram  (128 mels, FFT 1024, hop 512, 16 kHz)
      │
      ▼
convolutional autoencoder  (trained on normal clips only)
      │
      ▼
reconstruction error  (MSE between input and output)
      │
      ▼
anomaly score  →  NORMAL / ANOMALOUS verdict
```

High reconstruction error = the model has never seen sounds like this = likely anomalous.

---

## Dataset

[DCASE 2020 Task 2](https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds) — Unsupervised Anomalous Sound Detection in Machine Condition Monitoring.

| Machine | Train (normal) | Test (normal + anomalous) |
|---------|---------------|--------------------------|
| Fan     | ~1 000        | ~400 + 400               |
| Pump    | ~1 000        | ~400 + 400               |
| Valve   | ~1 000        | ~400 + 400               |

Download: [zenodo.org/records/3678171](https://zenodo.org/records/3678171) — see [`data/README.md`](data/README.md) for setup.

---

## Project Structure

```
auralytics/
├── src/
│   ├── preprocess.py   audio → log-mel spectrogram pipeline
│   ├── dataset.py      PyTorch Dataset + DataLoader factory
│   ├── model.py        convolutional autoencoder
│   ├── train.py        training loop with checkpointing
│   ├── evaluate.py     AUC-ROC, pAUC, F1 evaluation
│   └── utils.py        plotting, seeding, checkpointing
│
├── notebooks/
│   ├── 01_eda.ipynb          dataset inspection & spectrogram visualization
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb     train on Google Colab (free GPU)
│   └── 04_evaluation.ipynb   AUC curves, score distributions, failure cases
│
├── backend/
│   ├── main.py         FastAPI — POST /predict, GET /health
│   └── inference.py    model load + spectrogram → score pipeline
│
├── frontend/
│   ├── index.html      upload UI + score display
│   ├── style.css
│   └── app.js          fetch → backend, render spectrogram
│
├── data/               raw + processed audio (gitignored)
├── models/             trained .pth checkpoints (gitignored)
└── results/            figures, logs, evaluation outputs
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# See data/README.md for full instructions
# Download dev_data_fan.zip, dev_data_pump.zip, dev_data_valve.zip
# from https://zenodo.org/records/3678171
# Extract into data/raw/
```

### 3. Preprocess audio

```bash
python src/preprocess.py --machine_types fan pump valve
```

### 4. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Train (recommended: Google Colab free GPU)

```bash
python src/train.py --machine_type fan --epochs 50
```

### 6. Evaluate

```bash
python src/evaluate.py --machine_type fan
```

### 7. Run the web demo

```bash
# Backend
cd backend && uvicorn main:app --reload

# Frontend — open frontend/index.html in browser
```

---

## Evaluation Metrics

Following the official DCASE 2020 Task 2 protocol:

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Primary metric — threshold-free discrimination between normal and anomalous |
| **pAUC** | Partial AUC over FPR 0–0.1, penalizes false alarms |
| **F1** | At optimal threshold — balances precision and recall |

Target: AUC > 0.75 on all three machine types.

---

## Roadmap

- [x] Repo structure and data setup
- [x] Audio preprocessing pipeline (`preprocess.py`)
- [x] PyTorch Dataset class (`dataset.py`)
- [x] EDA notebook
- [ ] Convolutional autoencoder (`model.py`)
- [ ] Training loop (`train.py`)
- [ ] Evaluation pipeline (`evaluate.py`)
- [ ] FastAPI backend (`backend/`)
- [ ] Frontend demo (`frontend/`)
- [ ] Deployed live demo

---

## Built With

PyTorch · librosa · FastAPI · DCASE 2020

---

*CPEN 355 Final Project — UBC Electrical and Computer Engineering*
