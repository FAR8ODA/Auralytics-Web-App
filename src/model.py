"""
src/model.py
─────────────────────────────────────────────────────────────────────────────
Frame-level MLP Autoencoder for Auralytics.

Architecture
─────────────────────────────────────────────────────────────────────────────
Input: flattened window of N_FRAMES mel columns → (B, INPUT_DIM)
       default INPUT_DIM = 128 * 5 = 640

Encoder: INPUT_DIM → 512 → 256 → 128  (each: Linear + BN + ReLU + Dropout)
Bottleneck: 128-dim dense representation
Decoder: 128 → 256 → 512 → INPUT_DIM  (mirror, no activation on output layer)

Loss: MSE between input window and reconstruction.
Anomaly score per clip: mean reconstruction error across all windows from
that clip (see evaluate.py for aggregation).

Why this works better than the 2D conv AE
─────────────────────────────────────────────────────────────────────────────
The 2D conv AE treats the spectrogram as an image and generalizes too well —
it reconstructs both normal and anomalous clips similarly because it learns
global image statistics. The frame-level MLP instead learns the statistics of
short spectral patterns. Anomalous frames produce patterns the model has never
seen, leading to higher reconstruction error and better discrimination.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import INPUT_DIM


class _Block(nn.Module):
    """Linear → ReLU (no BatchNorm or Dropout — we want the AE to overfit to normal patterns)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPAutoencoder(nn.Module):
    """
    Frame-level MLP autoencoder.

    Architecture mirrors the DCASE 2020 Task 2 baseline:
        input → 128 → 128 → 128 → bottleneck → 128 → 128 → 128 → input

    Key design choices:
    - bottleneck=8 gives 80x compression (vs the old 128 which was only 5x).
      A tiny bottleneck forces the encoder to keep only the most common normal
      patterns, so anomalous frames that don't fit produce high reconstruction
      error and become detectable.
    - No BatchNorm or Dropout. These regularisers make the model generalise
      too well — they allow anomalous inputs to be reconstructed cleanly too,
      collapsing AUC toward 0.5.

    Args:
        input_dim  : flattened window size (N_MELS * N_FRAMES), default 640
        bottleneck : dimension of the latent code, default 8
    """

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        bottleneck: int = 8,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.bottleneck = bottleneck

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            _Block(input_dim,  128),
            _Block(128,        128),
            _Block(128,        128),
            _Block(128,        bottleneck),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            _Block(bottleneck, 128),
            _Block(128,        128),
            _Block(128,        128),
            nn.Linear(128, input_dim),   # no activation on output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → reconstruction: (B, input_dim)"""
        return self.decoder(self.encoder(x))

    def anomaly_score_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window reconstruction error, shape (B,).
        Use with torch.no_grad() at inference time.
        """
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MLPAutoencoder(\n"
            f"  input_dim  : {self.input_dim}\n"
            f"  encoder    : {self.input_dim} -> 128 -> 128 -> 128 -> {self.bottleneck}\n"
            f"  decoder    : {self.bottleneck} -> 128 -> 128 -> 128 -> {self.input_dim}\n"
            f"  params     : {self.count_parameters():,}\n)"
        )
