"""Inference engine for the Auralytics web demo.

The demo loads one selected per-ID MLP autoencoder per machine type:
- fan/id_06   AUC 0.879, threshold 0.44144
- pump/id_04  AUC 0.971, threshold 0.67290
- valve/id_04 AUC 0.763, threshold 0.32718

The preprocessing path intentionally matches src/preprocess.py and the final
Colab v2 training run: raw audio -> log-mel dB with ref=1.0, no per-clip
standardization, then global window normalization from the saved .npz file.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import librosa
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
DURATION = 10.0
N_FRAMES = 5
INPUT_DIM = N_MELS * N_FRAMES

MODEL_CONFIGS = {
    "fan": {
        "checkpoint": "fan_id_06_mlp_best.pth",
        "normalizer": "fan_id_06_normalizer.npz",
        "threshold": 0.44144,
        "label": "Fan (ID 06)",
        "auc": 0.879,
    },
    "pump": {
        "checkpoint": "pump_id_04_mlp_best.pth",
        "normalizer": "pump_id_04_normalizer.npz",
        "threshold": 0.67290,
        "label": "Pump (ID 04)",
        "auc": 0.971,
    },
    "valve": {
        "checkpoint": "valve_id_04_mlp_best.pth",
        "normalizer": "valve_id_04_normalizer.npz",
        "threshold": 0.32718,
        "label": "Valve (ID 04)",
        "auc": 0.763,
    },
}

MachineType = Literal["fan", "pump", "valve"]


class _Block(nn.Module):
    """Linear + ReLU block used by the final trained MLP checkpoints."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPAutoencoder(nn.Module):
    """Frame-level MLP AE: 640 -> 128 -> 128 -> 128 -> 8 -> ... -> 640."""

    def __init__(self, input_dim: int = INPUT_DIM, bottleneck: int = 8):
        super().__init__()
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


class ModelRegistry:
    """Loads and caches all demo models once at backend startup."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._models: dict[str, MLPAutoencoder] = {}
        self._normalizers: dict[str, dict[str, np.ndarray]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference device: {self.device}")

    def loaded_models(self) -> list[str]:
        return sorted(self._models.keys())

    def load_all(self) -> None:
        for machine in MODEL_CONFIGS:
            self.load(machine)

    def load(self, machine: str) -> None:
        if machine in self._models:
            return

        cfg = MODEL_CONFIGS[machine]
        ckpt_path = self.models_dir / cfg["checkpoint"]
        norm_path = self.models_dir / cfg["normalizer"]

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing {ckpt_path}. Copy {cfg['checkpoint']} into backend/models/."
            )
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Missing {norm_path}. Copy {cfg['normalizer']} into backend/models/."
            )

        model = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=8).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        self._models[machine] = model

        norm_data = np.load(norm_path)
        std = norm_data["std"].astype(np.float32)
        self._normalizers[machine] = {
            "mean": norm_data["mean"].astype(np.float32),
            "std": np.maximum(std, 1e-8),
        }
        print(
            f"  [{machine}] {cfg['label']} loaded "
            f"(AUC {cfg['auc']:.3f}, threshold {cfg['threshold']})"
        )

    def get(self, machine: str):
        self.load(machine)
        return self._models[machine], self._normalizers[machine]


def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Load uploaded bytes as mono 16 kHz audio, padded/truncated to 10 seconds."""
    buf = io.BytesIO(audio_bytes)
    wave, _ = librosa.load(buf, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    return wave[:target_len]


def compute_log_mel(wave: np.ndarray) -> np.ndarray:
    """Compute raw log-mel dB features. Do not normalize per clip."""
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def extract_windows(spec: np.ndarray) -> np.ndarray:
    """Flatten sliding 5-frame mel windows into shape (num_windows, 640)."""
    total_frames = spec.shape[1]
    if total_frames < N_FRAMES:
        raise ValueError(f"Spectrogram has only {total_frames} frames; need at least {N_FRAMES}.")
    windows = np.stack(
        [spec[:, start : start + N_FRAMES].flatten() for start in range(total_frames - N_FRAMES + 1)]
    )
    return windows.astype(np.float32)


def predict(audio_bytes: bytes, machine: MachineType, registry: ModelRegistry) -> dict:
    """Run full clip-level anomaly detection and return score plus PNG visualizations."""
    model, norm = registry.get(machine)
    cfg = MODEL_CONFIGS[machine]
    threshold = cfg["threshold"]

    wave = load_audio_bytes(audio_bytes)
    spec = compute_log_mel(wave)
    windows = extract_windows(spec)
    windows = (windows - norm["mean"]) / norm["std"]
    tensor = torch.from_numpy(windows).to(registry.device)

    with torch.no_grad():
        recon = model(tensor)
        errors = F.mse_loss(recon, tensor, reduction="none").mean(dim=1).cpu().numpy()

    score = float(errors.mean())
    verdict = "ANOMALOUS" if score >= threshold else "NORMAL"

    return {
        "score": round(score, 5),
        "verdict": verdict,
        "threshold": round(threshold, 5),
        "machine": machine,
        "label": cfg["label"],
        "auc": cfg["auc"],
        "spectrogram": _spectrogram_to_b64(spec, cfg["label"], score, verdict),
        "error_map": _error_map_to_b64(errors, threshold),
    }


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _spectrogram_to_b64(spec: np.ndarray, label: str, score: float, verdict: str) -> str:
    color = "#ef4444" if verdict == "ANOMALOUS" else "#34d399"
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    cb = plt.colorbar(im, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.label.set_color("#888")
    cb.ax.tick_params(colors="#555")
    ax.set_title(
        f"{label} - Score: {score:.5f} - {verdict}",
        color=color,
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Time frames", color="#888")
    ax.set_ylabel("Mel bin", color="#888")
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _error_map_to_b64(errors: np.ndarray, threshold: float) -> str:
    heat = np.tile(errors, (24, 1))
    fig, ax = plt.subplots(figsize=(10, 1.2))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(heat, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=threshold * 2)
    ax.set_title("Per-Window Reconstruction Error", color="#888", fontsize=9)
    ax.set_xlabel("Window index", color="#555", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors="#555", labelsize=7)
    plt.tight_layout()
    return _fig_to_b64(fig)
