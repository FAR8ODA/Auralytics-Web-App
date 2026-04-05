"""
src/utils.py
─────────────────────────────────────────────────────────────────────────────
Shared utilities for Auralytics: reproducibility, checkpointing, plotting.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save model + optimizer state to a .pth file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "loss":       loss,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load checkpoint from path into model (and optionally optimizer).
    Returns the checkpoint dict so callers can read epoch / loss.
    """
    device    = device or get_device()
    ckpt      = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return ckpt


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Log-Mel Spectrogram",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a single log-mel spectrogram.

    Args:
        spec      : 2-D array of shape (N_MELS, T)
        title     : plot title
        save_path : if given, save figure instead of showing it
    """
    if spec.ndim == 3:
        spec = spec[0]   # remove channel dim if present

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    anomaly_score: float,
    label: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Side-by-side plot of original vs reconstructed spectrogram
    with the reconstruction error map and anomaly score.
    """
    if original.ndim == 3:
        original      = original[0]
        reconstructed = reconstructed[0]

    error = np.abs(original - reconstructed)
    verdict = "ANOMALOUS" if label else "NORMAL"
    color   = "#ef4444"   if label else "#34d399"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Verdict: {verdict}  |  Anomaly Score: {anomaly_score:.4f}",
        fontsize=13, color=color, fontweight="bold"
    )

    for ax, data, title, cmap in zip(
        axes,
        [original, reconstructed, error],
        ["Input", "Reconstructed", "Error Map"],
        ["magma", "magma", "hot"],
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, format="%+2.0f dB")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_loss_curve(
    train_losses: list,
    val_losses: list,
    machine_type: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot training and validation reconstruction loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", linewidth=2)
    ax.plot(epochs, val_losses,   label="Val",   linewidth=2, linestyle="--")
    ax.set_title(f"Reconstruction Loss — {machine_type}", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_score_distribution(
    normal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
    machine_type: str,
    threshold: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Histogram of anomaly scores for normal vs anomalous clips."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores,    bins=40, alpha=0.65, label="Normal",    color="#34d399")
    ax.hist(anomalous_scores, bins=40, alpha=0.65, label="Anomalous", color="#ef4444")
    if threshold is not None:
        ax.axvline(threshold, color="white", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.4f}")
    ax.set_title(f"Anomaly Score Distribution — {machine_type}", fontsize=13)
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
