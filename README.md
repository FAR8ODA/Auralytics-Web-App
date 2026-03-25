# Auralytics

Acoustic anomaly detection for predictive maintenance.

Auralytics is a machine-learning web app that listens to machine audio, detects abnormal operating behavior, and visualizes why the clip looks anomalous. The project is built around the DCASE 2020 Task 2 benchmark for unsupervised anomalous sound detection in machine condition monitoring.

## Repository Purpose

This repository is structured to support the full pipeline:

- dataset setup and inspection
- audio preprocessing and spectrogram generation
- baseline and improved anomaly-detection models
- evaluation and error analysis
- a web demo for live presentation

## Canonical Repository Layout

The folders below are the intended project structure:

```text
Auralytics Web App/
|-- backend/           # FastAPI inference service
|-- data/              # Dataset instructions and local raw/processed data
|-- docs/              # Roadmap, demo spec, and project notes
|-- frontend/          # Presentation-facing frontend assets
|-- models/            # Saved checkpoints (gitignored)
|-- notebooks/         # Exploratory notebooks
|-- results/           # Figures, logs, exported evaluation outputs
|-- src/               # Training, preprocessing, dataset, and evaluation code
|-- requirements.txt   # Core project dependencies
`-- .gitignore
```

## Dataset

We will start with the official DCASE 2020 Task 2 machine-condition-monitoring data and focus first on:

- `fan`
- `pump`
- `valve`

See [data/README.md](C:/Users/farbo/Desktop/355/Final%20Project-Machine%20Anamoly%20Detector/Auralytics%20Web%20App/data/README.md) for the expected dataset layout and download notes.

## Final Demo Vision

The demo should let a user:

- upload a machine audio clip
- select the machine type
- receive an anomaly score and normal/anomalous verdict
- view a spectrogram and highlighted suspicious region
- compare against a known normal sample

See [docs/demo_spec.md](C:/Users/farbo/Desktop/355/Final%20Project-Machine%20Anamoly%20Detector/Auralytics%20Web%20App/docs/demo_spec.md) for the intended presentation flow.

## Immediate Next Steps

1. Download and unpack the DCASE data into `data/raw/`.
2. Implement dataset indexing and audio preprocessing in `src/`.
3. Train the baseline autoencoder.
4. Add backend inference.
5. Wire the frontend to the backend for the final live demo.

## Notes On Current Local State

- `frontend/` is the canonical UI directory.
- `backend/` is the canonical API directory.
- `web/` and the malformed brace-named scratch directory are treated as legacy local scratch space and are intentionally ignored from Git tracking.
