# src/dataset.py
# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset class for DCASE 2020 Task 2 audio clips.
#
# Responsibilities:
#   - Load .wav files from data/raw/{machine_type}/train|test/
#   - Return log-mel spectrogram tensors
#   - Provide labels (0=normal, 1=anomalous) for test set evaluation
#
# TODO: implement
# ──────────────────────────────────────────────────────────────────────────────
