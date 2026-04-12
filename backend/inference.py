"""
backend/inference.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Inference engine for the Auralytics demo.

Loads all three demo models on startup and caches them in memory:
  fan/id_06   AUC 0.879  threshold 0.44144
  pump/id_04  AUC 0.971  threshold 0.67290
  valve/id_04 AUC 0.763  threshold 0.32718

Pipeline per request:
  raw audio bytes
    â†’ log-mel spectrogram  (matches preprocess.py exactly â€” no per-clip norm)
    â†’ frame windows        (N_FRAMES=5, hop=1)
    â†’ global normalization (from saved .npz)
    â†’ MLP autoencoder      (from saved .pth)
    â†’ mean window MSE      â†’ clip score
    â†’ threshold compare    â†’ verdict
    â†’ spectrogram + error heatmap as base64 PNGs
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

import io
import base64
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# â”€â”€ Constants â€” must match preprocess.py and training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SAMPLE_RATE = 16_000
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
DURATION    = 10.0
N_FRAMES    = 5
INPUT_DIM   = N_MELS * N_FRAMES   # 640

# â”€â”€ Model configs â€” checkpoint name, normalizer name, eval threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_CONFIGS = {
    "fan": {
        "checkpoint":  "fan_id_06_mlp_best.pth",
        "normalizer":  "fan_id_06_normalizer.npz",
        "threshold":   0.44144,
        "label":       "Fan (ID 06)",
        "auc":         0.879,
    },
    "pump": {
        "checkpoint":  "pump_id_04_mlp_best.pth",
        "normalizer":  "pump_id_04_normalizer.npz",
        "threshold":   0.67290,
        "label":       "Pump (ID 04)",
        "auc":         0.971,
    },
    "valve": {
        "checkpoint":  "valve_id_04_mlp_best.pth",
        "normalizer":  "valve_id_04_normalizer.npz",
        "threshold":   0.32718,
        "label":       "Valve (ID 04)",
        "auc":         0.763,
    },
}

MachineType = Literal["fan", "pump", "valve"]


# â”€â”€ MLP Autoencoder â€” must exactly mirror src/model.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.block(x)


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            _Block(input_dim, 128), _Block(128, 128),
            _Block(128, 128),       _Block(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            _Block(bottleneck, 128), _Block(128, 128),
            _Block(128, 128),        nn.Linear(128, input_dim),
        )
    def forward(self, x): return self.decoder(self.encoder(x))


# â”€â”€ Model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ModelRegistry:
    """Loads and caches all three models at startup."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._models:      dict[str, MLPAutoencoder] = {}
        self._normalizers: dict[str, dict]           = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference device: {self.device}")

    def load_all(self) -> None:
        for machine, cfg in MODEL_CONFIGS.items():
            ckpt_path = self.models_dir / cfg["checkpoint"]
            norm_path = self.models_dir / cfg["normalizer"]

            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing: {ckpt_path}\n"
                    f"Download {cfg['checkpoint']} from Colab Drive â†’ backend/models/"
                )
            if not norm_path.exists():
                raise FileNotFoundError(
                    f"Missing: {norm_path}\n"
                    f"Download {cfg['normalizer']} from Colab Drive â†’ backend/models/"
                )

            model = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=8).to(self.device)
            ckpt  = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self._models[machine] = model

            norm_data = np.load(norm_path)
            self._normalizers[machine] = {
                "mean": norm_data["mean"].astype(np.float32),
                "std":  norm_data["std"].astype(np.float32),
            }
            print(f"  [{machine}] {cfg['label']} loaded  "
                  f"(AUC {cfg['auc']:.3f}, threshold {cfg['threshold']})")

    def get(self, machine: str):
        return self._models[machine], self._normalizers[machine]


# â”€â”€ Audio pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    buf  = io.BytesIO(audio_bytes)
    wave, _ = librosa.load(buf, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    n_target = int(SAMPLE_RATE * DURATION)
    if len(wave) < n_target:
        wave = np.pad(wave, (0, n_target - len(wave)))
    return wave[:n_target]


def compute_log_mel(wave: np.ndarray) -> np.ndarray:
    """No per-clip normalization â€” global window norm is applied later."""
    mel = librosa.feature.melspectrogram(
        y=wave, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def extract_windows(spec: np.ndarray) -> np.ndarray:
    T       = spec.shape[1]
    starts  = range(0, T - N_FRAMES + 1, 1)
    windows = np.stack([spec[:, t:t + N_FRAMES].flatten() for t in starts])
    return windows.astype(np.float32)


# â”€â”€ Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def predict(audio_bytes: bytes, machine: MachineType, registry: ModelRegistry) -> dict:
    """
    Full inference for one clip. Returns score, verdict, and visualization PNGs.
    """
    model, norm = registry.get(machine)
    device      = registry.device
    cfg         = MODEL_CONFIGS[machine]
    threshold   = cfg["threshold"]

    wave    = load_audio_bytes(audio_bytes)
    spec    = compute_log_mel(wave)

    windows = extract_windows(spec)
    windows = (windows - norm["mean"]) / norm["std"]
    tensor  = torch.from_numpy(windows).to(device)

    with torch.no_grad():
        recon  = model(tensor)
        errors = F.mse_loss(recon, tensor, reduction="none").mean(dim=1).cpu().numpy()

    score   = float(errors.mean())
    verdict = "ANOMALOUS" if score >= threshold else "NORMAL"

    return {
        "score":       round(score, 5),
        "verdict":     verdict,
        "threshold":   round(threshold, 5),
        "machine":     machine,
        "label":       cfg["label"],
        "auc":         cfg["auc"],
        "spectrogram": _spectrogram_to_b64(spec, cfg["label"], score, verdict),
        "error_map":   _error_map_to_b64(errors, threshold),
    }


# â”€â”€ Visualizations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _spectrogram_to_b64(spec, label, score, verdict) -> str:
    color = "#ef4444" if verdict == "ANOMALOUS" else "#34d399"
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    cb = plt.colorbar(im, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.label.set_color("#888")
    cb.ax.tick_params(colors="#555")
    ax.set_title(
        f"{label}  Â·  Score: {score:.5f}  Â·  {verdict}",
        color=color, fontsize=11, fontweight="bold", pad=8
    )
    ax.set_xlabel("Time frames", color="#888")
    ax.set_ylabel("Mel bin",     color="#888")
    ax.tick_params(colors="#555")
    for s in ax.spines.values(): s.set_edgecolor("#333")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _error_map_to_b64(errors, threshold) -> str:
    heat = np.tile(errors, (24, 1))
    fig, ax = plt.subplots(figsize=(10, 1.2))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(heat, aspect="auto", origin="lower",
                   cmap="hot", vmin=0, vmax=threshold * 2)
    ax.set_title("Per-Window Reconstruction Error", color="#888", fontsize=9)
    ax.set_xlabel("Window index", color="#555", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(colors="#555", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#333")
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors="#555", labelsize=7)
    plt.tight_layout()
    return _fig_to_b64(fig)

