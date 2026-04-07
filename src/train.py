"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop for the Auralytics MLP frame-level autoencoder.

Key differences from the 2D conv AE training:
  - Unit of training is a window of N_FRAMES mel columns, not a full clip
  - Global normalizer is fitted on training windows and saved alongside the
    checkpoint — it MUST be reused at eval time (no data leakage)
  - Much larger effective dataset (thousands of windows per clip)
  - Larger batch size (512) works well for MLP on frame data

Usage:
    python -m src.train --machine_type fan
    python -m src.train --machine_type fan --epochs 50 --batch_size 512 --lr 1e-3

Outputs:
    models/{machine_type}_mlp_best.pth        model checkpoint
    models/{machine_type}_normalizer.npz      global normalizer (required for eval)
    results/{machine_type}_mlp_loss.png       loss curve
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import make_train_val_loaders, N_FRAMES, INPUT_DIM
from src.model import MLPAutoencoder
from src.utils import get_device, plot_loss_curve, save_checkpoint, set_seed

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")


def run_epoch(
    model:     nn.Module,
    loader,
    optimizer,
    device:    torch.device,
    train:     bool,
) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for windows, _ in loader:
            windows = windows.to(device)
            recon   = model(windows)
            loss    = nn.functional.mse_loss(recon, windows)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(windows)

    return total_loss / len(loader.dataset)


def train_model(
    machine_type:  str,
    processed_dir: Path = PROCESSED_DIR,
    models_dir:    Path = MODELS_DIR,
    results_dir:   Path = RESULTS_DIR,
    epochs:        int   = 50,
    batch_size:    int   = 512,
    lr:            float = 1e-3,
    patience:      int   = 8,
    n_frames:      int   = N_FRAMES,
    seed:          int   = 42,
) -> None:
    set_seed(seed)
    device = get_device()

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data + normalizer ─────────────────────────────────────────────────────
    print(f"\nLoading data for: {machine_type}")
    train_loader, val_loader, normalizer = make_train_val_loaders(
        processed_dir, machine_type,
        batch_size=batch_size, n_frames=n_frames,
    )

    # Save normalizer immediately — needed at eval time
    norm_path = models_dir / f"{machine_type}_normalizer.npz"
    normalizer.save(norm_path)
    print(f"  Normalizer saved: {norm_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=128).to(device)
    print(f"\n{model}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    ckpt_path = models_dir / f"{machine_type}_mlp_best.pth"

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            tag = "saved"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve {epochs_no_improve}/{patience})"

        print(f"Epoch {epoch:03d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | {tag}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Normalizer    : {norm_path}")

    plot_loss_curve(
        train_losses, val_losses, machine_type,
        save_path=results_dir / f"{machine_type}_mlp_loss.png",
    )
    print(f"Loss curve    : {results_dir / f'{machine_type}_mlp_loss.png'}")


def main():
    parser = argparse.ArgumentParser(description="Auralytics MLP AE — train")
    parser.add_argument("--machine_type",  required=True, choices=["fan", "pump", "valve"])
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir",    type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir",   type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=8)
    parser.add_argument("--n_frames",      type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    train_model(
        machine_type  = args.machine_type,
        processed_dir = args.processed_dir,
        models_dir    = args.models_dir,
        results_dir   = args.results_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        patience      = args.patience,
        n_frames      = args.n_frames,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
