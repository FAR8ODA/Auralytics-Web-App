# src/preprocess.py
# ──────────────────────────────────────────────────────────────────────────────
# Audio preprocessing utilities.
#
# Responsibilities:
#   - Load raw .wav files with librosa
#   - Compute log-mel spectrograms (128 mel bins, FFT=1024, hop=512)
#   - Normalize per-clip amplitude
#   - Save processed spectrograms to data/processed/
#
# TODO: implement
# ──────────────────────────────────────────────────────────────────────────────
