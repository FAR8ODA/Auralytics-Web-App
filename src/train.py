# src/train.py
# ──────────────────────────────────────────────────────────────────────────────
# Training loop for the convolutional autoencoder.
#
# Responsibilities:
#   - Accept CLI args: --machine_type, --epochs, --lr, --batch_size
#   - Load training dataset (normal clips only)
#   - Train autoencoder with MSE reconstruction loss
#   - Save best model checkpoint to models/{machine_type}_best.pth
#   - Log loss curve to results/
#
# TODO: implement
# ──────────────────────────────────────────────────────────────────────────────
