# src/model.py
# ──────────────────────────────────────────────────────────────────────────────
# Convolutional Autoencoder for anomaly detection.
#
# Architecture:
#   Encoder: Conv2d → ReLU → MaxPool (x3 layers) → bottleneck
#   Decoder: ConvTranspose2d → ReLU (x3 layers) → reconstructed spectrogram
#
# Anomaly score = MSE(input_spectrogram, reconstructed_spectrogram)
# High reconstruction error → anomalous clip
#
# TODO: implement
# ──────────────────────────────────────────────────────────────────────────────
