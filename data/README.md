# Data Setup

This folder will hold the DCASE 2020 Task 2 dataset used by Auralytics.

Raw audio is intentionally gitignored. Download it manually and keep the original folder names.

## Machine Types In Initial Scope

We will start with:

- `fan`
- `pump`
- `valve`

## Required Download

Development dataset:

- https://zenodo.org/records/3678171

Download the archives for the machine types above and extract them into `data/raw/`.

## Optional Download

Additional training dataset for later improvement:

- https://zenodo.org/records/3727685

## Expected Folder Layout

```text
data/
|-- raw/
|   |-- fan/
|   |   |-- train/      # normal clips only
|   |   `-- test/       # normal + anomalous clips
|   |-- pump/
|   `-- valve/
`-- processed/          # generated later by src/preprocess.py
```

## DCASE Naming Convention

- normal clip: `normal_id_00_00000000.wav`
- anomalous clip: `anomaly_id_00_00000000.wav`

Each clip is approximately 10 seconds, mono, 16 kHz.
