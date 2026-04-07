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
    """Linear → BatchNorm → ReLU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPAutoencoder(nn.Module):
    """
    Frame-level MLP autoencoder.

    Args:
        input_dim  : flattened window size (N_MELS * N_FRAMES), default 640
        bottleneck : dimension of the latent code, default 128
        dropout    : dropout probability in encoder/decoder blocks
    """

    def __init__(
        self,
        input_dim:  int   = INPUT_DIM,
        bottleneck: int   = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.bottleneck = bottleneck

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            _Block(input_dim,  512, dropout),
            _Block(512,        256, dropout),
            _Block(256,        bottleneck, dropout),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            _Block(bottleneck, 256, dropout),
            _Block(256,        512, dropout),
            nn.Linear(512, input_dim),   # no BN/ReLU/Dropout on output
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
            f"  encoder    : {self.input_dim} -> 512 -> 256 -> {self.bottleneck}\n"
            f"  decoder    : {self.bottleneck} -> 256 -> 512 -> {self.input_dim}\n"
            f"  params     : {self.count_parameters():,}\n)"
        )
