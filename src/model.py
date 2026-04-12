"""Frame-level MLP autoencoder used by the final Auralytics pipeline.

Input windows are five adjacent log-mel frames flattened into a 640-dimensional
vector. A small bottleneck forces the model to reconstruct normal machine
patterns well while giving higher reconstruction error on unfamiliar sounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import INPUT_DIM


class _Block(nn.Module):
    """Linear + ReLU block. No BatchNorm or Dropout for this final AE."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPAutoencoder(nn.Module):
    """MLP AE: input -> 128 -> 128 -> 128 -> bottleneck -> 128 -> 128 -> 128 -> input."""

    def __init__(self, input_dim: int = INPUT_DIM, bottleneck: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck = bottleneck

        self.encoder = nn.Sequential(
            _Block(input_dim, 128),
            _Block(128, 128),
            _Block(128, 128),
            _Block(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            _Block(bottleneck, 128),
            _Block(128, 128),
            _Block(128, 128),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Return one reconstruction MSE per input window."""
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            "MLPAutoencoder(\n"
            f"  input_dim  : {self.input_dim}\n"
            f"  encoder    : {self.input_dim} -> 128 -> 128 -> 128 -> {self.bottleneck}\n"
            f"  decoder    : {self.bottleneck} -> 128 -> 128 -> 128 -> {self.input_dim}\n"
            f"  params     : {self.count_parameters():,}\n)"
        )
