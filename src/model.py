"""
src/model.py
-----------------------------------------------------------------------------
Convolutional Autoencoder for Auralytics.

Architecture overview
---------------------
Input: (B, 1, 128, T) - batch of log-mel spectrograms

Encoder
  Conv2d(1->32)   + BN + ReLU + MaxPool(2,2) -> (B, 32, 64, T/2)
  Conv2d(32->64)  + BN + ReLU + MaxPool(2,2) -> (B, 64, 32, T/4)
  Conv2d(64->128) + BN + ReLU + MaxPool(2,2) -> (B, 128, 16, T/8)

Decoder (Upsample + Conv avoids checkerboard artifacts from ConvTranspose2d)
  Upsample(x2) + Conv2d(128->64) + BN + ReLU -> (B, 64, 32, T/4)
  Upsample(x2) + Conv2d(64->32)  + BN + ReLU -> (B, 32, 64, T/2)
  Upsample(x2) + Conv2d(32->1)                 -> (B, 1, 128, T) [resized to match input]

Anomaly score
  MSE(input, reconstruction) averaged over (C, H, W) -> scalar per clip.
  Normal clips reconstruct accurately -> low score.
  Anomalous clips reconstruct poorly  -> high score.
-----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Upsample(x2) -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, activation: bool = True):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
        ]
        if activation:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for unsupervised anomaly detection."""

    def __init__(self, base_ch: int = 32):
        super().__init__()

        self.enc1 = _EncoderBlock(1, base_ch)
        self.enc2 = _EncoderBlock(base_ch, base_ch * 2)
        self.enc3 = _EncoderBlock(base_ch * 2, base_ch * 4)

        self.dec1 = _DecoderBlock(base_ch * 4, base_ch * 2)
        self.dec2 = _DecoderBlock(base_ch * 2, base_ch)
        self.dec3 = _DecoderBlock(base_ch, 1, activation=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc3(self.enc2(self.enc1(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec3(self.dec2(self.dec1(z)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z)
        # MaxPool floors odd widths (e.g. 313 -> 156 -> 78 -> 39), so three
        # x2 upsampling stages only recover 312. Interpolate guarantees the
        # output matches the original spatial size exactly.
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
        return recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-clip reconstruction MSE, shape (B,)."""
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ConvAutoencoder(\n"
            f"  encoder: 1->{self.enc1.block[0].out_channels}"
            f"->{self.enc2.block[0].out_channels}"
            f"->{self.enc3.block[0].out_channels}\n"
            f"  decoder: mirrors encoder\n"
            f"  params:  {self.count_parameters():,}\n)"
        )
