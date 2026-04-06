"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop for the Auralytics convolutional autoencoder.

The model is trained to reconstruct log-mel spectrograms of NORMAL clips.
No anomalous data is used during training (fully unsupervised).

Loss: MSE between input spectrogram and reconstruction.

Usage:
    python -m src.train --machine_type fan
    python -m src.train --machine_type pump --epochs 60 --lr 1e-3 --batch_size 64

Outputs:
    models/{machine_type}_best.pth   ← best checkpoint by val loss
    results/{machine_type}_loss.png  ← loss curve
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import make_dataloaders
from src.model import ConvAutoencoder
from src.utils import get_device, plot_loss_curve, save_checkpoint, set_seed


# ── Defaults ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    train: bool,
) -> float:
    """Run one epoch. Returns mean MSE loss."""
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for specs, _ in loader:           # labels unused during training
            specs = specs.to(device)
            recon = model(specs)
            loss  = nn.functional.mse_loss(recon, specs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(specs)

    return total_loss / len(loader.dataset)


def train_model(
    machine_type: str,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path    = MODELS_DIR,
    results_dir: Path   = RESULTS_DIR,
    epochs: int         = 50,
    batch_size: int     = 32,
    lr: float           = 1e-3,
    patience: int       = 8,
    seed: int           = 42,
) -> None:
    """
    Full training run for one machine type.

    Early stopping halts training if val loss does not improve for
    `patience` consecutive epochs. Best checkpoint is always kept.
    """
    set_seed(seed)
    device = get_device()

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\nLoading data for: {machine_type}")
    train_loader, val_loader, _ = make_dataloaders(
        processed_dir, machine_type, batch_size=batch_size
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConvAutoencoder(base_ch=32).to(device)
    print(f"\n{model}")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    ckpt_path = models_dir / f"{machine_type}_best.pth"

    print(f"\nTraining for up to {epochs} epochs (early stopping patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            tag = "saved"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve {epochs_no_improve}/{patience})"

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | {tag}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Post-training ─────────────────────────────────────────────────────────
    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {ckpt_path}")

    plot_loss_curve(
        train_losses, val_losses, machine_type,
        save_path=results_dir / f"{machine_type}_loss.png",
    )
    print(f"Loss curve    : {results_dir / f'{machine_type}_loss.png'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auralytics — train autoencoder")
    parser.add_argument("--machine_type",  required=True, choices=["fan", "pump", "valve"])
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir",    type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir",   type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=8)
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
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
