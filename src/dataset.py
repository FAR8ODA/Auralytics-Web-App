"""
src/dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset for Auralytics.

Loads pre-processed log-mel spectrograms (.npy) from data/processed/.
Training set contains normal clips only (unsupervised setting).
Test set contains both normal and anomalous clips with ground-truth labels.

Label convention:
    0 → normal
    1 → anomalous
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _label_from_filename(filename: str) -> int:
    """
    Parse DCASE filename convention.
    Files starting with 'anomaly_' are labelled 1, everything else 0.
    """
    return 1 if filename.startswith("anomaly") else 0


class DCASEDataset(Dataset):
    """
    Dataset for a single machine type and split.

    Args:
        processed_dir : path to data/processed/
        machine_type  : one of 'fan', 'pump', 'valve'
        split         : 'train' or 'test'
        transform     : optional callable applied to the spectrogram tensor
    """

    def __init__(
        self,
        processed_dir: Path,
        machine_type: str,
        split: str = "train",
        transform=None,
    ):
        self.split_dir   = Path(processed_dir) / machine_type / split
        self.transform   = transform
        self.machine     = machine_type
        self.split       = split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found at {self.split_dir}. "
                f"Run: python src/preprocess.py --machine_types {machine_type}"
            )

        self.files = sorted(self.split_dir.glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.split_dir}")

        # Labels — 0 for training (all normal), parsed from name for test
        self.labels = [_label_from_filename(f.stem) for f in self.files]

        # Sanity check: train split should have no anomalies
        if split == "train":
            n_anomalous = sum(self.labels)
            if n_anomalous > 0:
                print(f"[warn] {n_anomalous} anomalous clips found in train split — check your data layout")

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        spec = np.load(self.files[idx])                     # (N_MELS, T)
        spec = torch.from_numpy(spec).unsqueeze(0)          # (1, N_MELS, T)

        if self.transform:
            spec = self.transform(spec)

        label = self.labels[idx]
        return spec, label

    # ── Convenience helpers ───────────────────────────────────────────────

    @property
    def spec_shape(self):
        """Return shape of a single spectrogram (C, H, W)."""
        spec, _ = self[0]
        return tuple(spec.shape)

    def class_counts(self) -> dict:
        """Return count of normal and anomalous clips."""
        n_anomalous = sum(self.labels)
        n_normal    = len(self.labels) - n_anomalous
        return {"normal": n_normal, "anomalous": n_anomalous}

    def __repr__(self) -> str:
        counts = self.class_counts()
        return (
            f"DCASEDataset(machine={self.machine}, split={self.split}, "
            f"n={len(self)}, normal={counts['normal']}, anomalous={counts['anomalous']})"
        )


def make_dataloaders(
    processed_dir: Path,
    machine_type: str,
    batch_size: int = 32,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Build train, validation, and test DataLoaders for one machine type.

    Validation is carved out of the train split (normal clips only).
    Test loader loads the full test split with labels for evaluation.

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, random_split

    full_train = DCASEDataset(processed_dir, machine_type, split="train")
    test_ds    = DCASEDataset(processed_dir, machine_type, split="test")

    # Train / val split
    n_val   = max(1, int(len(full_train) * val_fraction))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
