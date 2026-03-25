# src/evaluate.py
# ──────────────────────────────────────────────────────────────────────────────
# Evaluation pipeline for trained models.
#
# Responsibilities:
#   - Load trained model checkpoint
#   - Run inference on test set (normal + anomalous clips)
#   - Compute anomaly scores (reconstruction MSE per clip)
#   - Calculate AUC-ROC, pAUC (FPR 0-0.1), F1 at optimal threshold
#   - Output per-machine-type results table
#   - Save ROC curve plots to results/figures/
#
# TODO: implement
# ──────────────────────────────────────────────────────────────────────────────
