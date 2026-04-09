"""Training loop for the Auralytics frame-level MLP autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import INPUT_DIM, N_FRAMES, available_machine_ids, make_train_val_loaders
from src.model import MLPAutoencoder
from src.utils import get_device, plot_loss_curve, save_checkpoint, set_seed

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def run_tag(machine_type: str, machine_id: Optional[str] = None) -> str:
    return f"{machine_type}_{machine_id}" if machine_id else machine_type


def scope_label(machine_type: str, machine_id: Optional[str] = None) -> str:
    return f"{machine_type} ({machine_id})" if machine_id else machine_type


<<<<<<< HEAD
def resolve_machine_ids(processed_dir: Path, machine_type: str, machine_id: Optional[str], all_ids: bool) -> list[Optional[str]]:
    ids = available_machine_ids(processed_dir, machine_type, split="train")
    if all_ids:
        if not ids:
            raise SystemExit(f"No machine IDs found for {machine_type} in {processed_dir}")
        return ids
    if machine_id is None:
        return [None]
    if machine_id not in ids:
        available = ", ".join(ids) if ids else "none"
        raise SystemExit(f"Unknown machine_id {machine_id!r} for {machine_type}. Available IDs: {available}")
    return [machine_id]


=======
>>>>>>> 2cc3d3d199f09874a4c02662066b50dad471aa0c
def run_epoch(model: nn.Module, loader, optimizer, device: torch.device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for windows, _ in loader:
            windows = windows.to(device)
            recon = model(windows)
            loss = nn.functional.mse_loss(recon, windows)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(windows)
    return total_loss / len(loader.dataset)


def train_model(
    machine_type: str,
    machine_id: Optional[str] = None,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 8,
    n_frames: int = N_FRAMES,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = get_device()
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data for: {scope_label(machine_type, machine_id)}")
    train_loader, val_loader, normalizer = make_train_val_loaders(
        processed_dir,
        machine_type,
        machine_id=machine_id,
        batch_size=batch_size,
        n_frames=n_frames,
        seed=seed,
    )

    tag = run_tag(machine_type, machine_id)
    norm_path = models_dir / f"{tag}_normalizer.npz"
    ckpt_path = models_dir / f"{tag}_mlp_best.pth"
    loss_path = results_dir / f"{tag}_mlp_loss.png"

    normalizer.save(norm_path)
    print(f"  Normalizer saved: {norm_path}")

    model = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=128).to(device)
    print(f"\n{model}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            tag_text = "saved"
        else:
            epochs_no_improve += 1
            tag_text = f"  (no improve {epochs_no_improve}/{patience})"

        print(f"Epoch {epoch:03d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | {tag_text}")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Normalizer    : {norm_path}")
    plot_loss_curve(train_losses, val_losses, tag, save_path=loss_path)
    print(f"Loss curve    : {loss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auralytics MLP AE - train")
    parser.add_argument("--machine_type", required=True, choices=["fan", "pump", "valve"])
    parser.add_argument("--machine_id", default=None, help="Optional machine ID such as id_00")
<<<<<<< HEAD
    parser.add_argument("--all_ids", action="store_true", help="Train one model per available machine ID")
    parser.add_argument("--list_ids", action="store_true", help="Print available machine IDs and exit")
=======
>>>>>>> 2cc3d3d199f09874a4c02662066b50dad471aa0c
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

<<<<<<< HEAD
    ids = available_machine_ids(args.processed_dir, args.machine_type, split="train")
    if args.list_ids:
        print("Available IDs:")
        for machine_id in ids:
            print(f"  {machine_id}")
        return
    if args.machine_id and args.all_ids:
        raise SystemExit("Use either --machine_id or --all_ids, not both")

    targets = resolve_machine_ids(args.processed_dir, args.machine_type, args.machine_id, args.all_ids)
    for target_id in targets:
        train_model(
            machine_type=args.machine_type,
            machine_id=target_id,
            processed_dir=args.processed_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            n_frames=args.n_frames,
            seed=args.seed,
        )
=======
    if args.machine_id and args.machine_id not in available_machine_ids(args.processed_dir, args.machine_type, split="train"):
        ids = ", ".join(available_machine_ids(args.processed_dir, args.machine_type, split="train"))
        raise SystemExit(f"Unknown machine_id {args.machine_id!r} for {args.machine_type}. Available IDs: {ids}")

    train_model(
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        n_frames=args.n_frames,
        seed=args.seed,
    )
>>>>>>> 2cc3d3d199f09874a4c02662066b50dad471aa0c


if __name__ == "__main__":
    main()
