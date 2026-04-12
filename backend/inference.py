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

import os
import threading

import base64
import io
import struct
import zlib
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
try:
    torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
except RuntimeError:
    pass


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
        self._lock = threading.Lock()
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

        with self._lock:
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
    """Load uploaded WAV bytes as mono 16 kHz audio, padded/truncated to 10 seconds."""
    wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        wave = _linear_resample(wave, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    return wave[:target_len].astype(np.float32)


def _linear_resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave.astype(np.float32)
    duration = len(wave) / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    old_x = np.linspace(0.0, duration, num=len(wave), endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, wave).astype(np.float32)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mask = hz >= min_log_hz
    if np.any(mask):
        mels = mels.copy()
        mels[mask] = min_log_mel + np.log(hz[mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    freqs = f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mask = mel >= min_log_mel
    freqs = np.where(mask, min_log_hz * np.exp(logstep * (mel - min_log_mel)), freqs)
    return freqs


def _mel_filterbank() -> np.ndarray:
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(SAMPLE_RATE / 2.0)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    mel_f = _mel_to_hz(mel_points)
    fftfreqs = np.linspace(0.0, SAMPLE_RATE / 2.0, 1 + N_FFT // 2)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    filters = np.zeros((N_MELS, len(fftfreqs)), dtype=np.float64)
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        filters[i] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : N_MELS + 2] - mel_f[:N_MELS])
    filters *= enorm[:, np.newaxis]
    return filters.astype(np.float32)


_MEL_FILTERBANK = _mel_filterbank()
_HANN_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def compute_log_mel(wave: np.ndarray) -> np.ndarray:
    """Compute raw log-mel dB features without runtime librosa dependencies."""
    padded = np.pad(wave.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="constant")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    starts = np.arange(n_frames)[:, None] * HOP_LENGTH + np.arange(N_FFT)[None, :]
    frames = padded[starts] * _HANN_WINDOW[None, :]
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (np.abs(spectrum) ** 2).T.astype(np.float32)
    mel = np.maximum(_MEL_FILTERBANK @ power, 1e-10)
    return (10.0 * np.log10(mel)).astype(np.float32)


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

def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack("!I", len(payload)) + chunk_type + payload + struct.pack("!I", crc)


def _rgb_png_to_b64(rgb: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB array as PNG using only the stdlib."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB image.")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(int)
    return image[np.ix_(y_idx, x_idx)]


def _industrial_colormap(values: np.ndarray) -> np.ndarray:
    """Small magma-like RGB map for fast spectrogram previews."""
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    r = np.clip(4.2 * values - 0.45, 0, 1)
    g = np.clip(3.2 * values - 1.15, 0, 1)
    b = np.clip(2.8 * (1.0 - values) + 0.25 * values, 0, 1) * (1.0 - 0.35 * values)
    warm = np.stack([r, g, b], axis=-1)
    return (warm * 255).astype(np.uint8)


def _hot_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    r = np.clip(3.0 * values, 0, 1)
    g = np.clip(3.0 * values - 1.0, 0, 1)
    b = np.clip(3.0 * values - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _spectrogram_to_b64(spec: np.ndarray, label: str, score: float, verdict: str) -> str:
    # Fast production preview: normalize dB values and encode directly as PNG.
    del label, score, verdict
    preview = np.flipud(spec)
    lo, hi = np.percentile(preview, [2, 98])
    norm = (preview - lo) / max(float(hi - lo), 1e-6)
    norm = _resize_nearest(norm, height=220, width=760)
    return _rgb_png_to_b64(_industrial_colormap(norm))


def _error_map_to_b64(errors: np.ndarray, threshold: float) -> str:
    norm = errors / max(float(threshold * 2.0), 1e-6)
    heat = np.tile(norm[None, :], (64, 1))
    heat = _resize_nearest(heat, height=80, width=760)
    return _rgb_png_to_b64(_hot_colormap(heat))
