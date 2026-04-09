"""PyTorch datasets and helpers for the Auralytics frame-level MLP autoencoder."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

N_FRAMES = 5
HOP = 1
N_MELS = 128
INPUT_DIM = N_MELS * N_FRAMES
_ID_RE = re.compile(r"(id_\d\d)")


def _label_from_filename(stem: str) -> int:
    return 1 if stem.startswith("anomaly") else 0


def extract_machine_id(stem: str) -> Optional[str]:
    match = _ID_RE.search(stem)
    return match.group(1) if match else None


def _select_files(files: list[Path], machine_id: Optional[str]) -> list[Path]:
    if machine_id is None:
        return files
    selected = [path for path in files if extract_machine_id(path.stem) == machine_id]
    return selected


def available_machine_ids(processed_dir: Path, machine_type: str, split: str = "train") -> list[str]:
    split_dir = Path(processed_dir) / machine_type / split
    files = sorted(split_dir.glob("*.npy"))
    ids = sorted({extract_machine_id(path.stem) for path in files if extract_machine_id(path.stem)})
    return ids


def extract_windows(spec: np.ndarray, n_frames: int = N_FRAMES, hop: int = HOP) -> np.ndarray:
    n_mels, total_frames = spec.shape
    if n_mels != N_MELS:
        raise ValueError(f"Expected {N_MELS} mel bins, got {n_mels}")
    if total_frames < n_frames:
        raise ValueError(f"Spectrogram has only {total_frames} frames, cannot extract windows of {n_frames}")
    starts = range(0, total_frames - n_frames + 1, hop)
    windows = np.stack([spec[:, start : start + n_frames].reshape(-1) for start in starts], axis=0)
    return windows.astype(np.float32)


class Normalizer:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit_array(self, data: np.ndarray) -> "Normalizer":
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = (data.std(axis=0) + 1e-8).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before transform()")
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted normalizer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        data = np.load(path)
        norm = cls()
        norm.mean = data["mean"]
        norm.std = data["std"]
        return norm


class FrameDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        machine_type: str,
        split: str = "train",
        machine_id: Optional[str] = None,
        n_frames: int = N_FRAMES,
        hop: int = HOP,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        split_dir = Path(processed_dir) / machine_type / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found at {split_dir}. Run: python -m src.preprocess --machine_types {machine_type}"
            )

        files = sorted(split_dir.glob("*.npy"))
        files = _select_files(files, machine_id)
        if not files:
            ids = available_machine_ids(processed_dir, machine_type, split=split)
            hint = f" Available IDs: {', '.join(ids)}" if ids else ""
            raise RuntimeError(f"No .npy files found in {split_dir} for machine_id={machine_id!r}.{hint}")

        self.machine_type = machine_type
        self.machine_id = machine_id
        self.normalizer = normalizer
        self.files = files
        self.num_clips = len(files)
        self._all = np.concatenate(
            [extract_windows(np.load(path).astype(np.float32), n_frames=n_frames, hop=hop) for path in files],
            axis=0,
        )

    def __len__(self) -> int:
        return len(self._all)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window = self._all[idx]
        if self.normalizer is not None:
            window = self.normalizer.transform(window)
        return torch.from_numpy(window), 0

    @property
    def input_dim(self) -> int:
        return self._all.shape[1]


class ClipDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        machine_type: str,
        split: str = "test",
        machine_id: Optional[str] = None,
    ) -> None:
        split_dir = Path(processed_dir) / machine_type / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Processed data not found at {split_dir}")

        files = sorted(split_dir.glob("*.npy"))
        files = _select_files(files, machine_id)
        if not files:
            ids = available_machine_ids(processed_dir, machine_type, split=split)
            hint = f" Available IDs: {', '.join(ids)}" if ids else ""
            raise RuntimeError(f"No .npy files found in {split_dir} for machine_id={machine_id!r}.{hint}")

        self.machine_type = machine_type
        self.machine_id = machine_id
        self.files = files
        self.labels = [_label_from_filename(path.stem) for path in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        spec = np.load(self.files[idx]).astype(np.float32)
        return spec, self.labels[idx]

    def class_counts(self) -> dict[str, int]:
        n_anomalous = sum(self.labels)
        return {"normal": len(self.labels) - n_anomalous, "anomalous": n_anomalous}


def make_train_val_loaders(
    processed_dir: Path,
    machine_type: str,
    machine_id: Optional[str] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 42,
    n_frames: int = N_FRAMES,
    hop: int = HOP,
):
    from torch.utils.data import DataLoader, random_split

    raw_ds = FrameDataset(
        processed_dir,
        machine_type,
        split="train",
        machine_id=machine_id,
        n_frames=n_frames,
        hop=hop,
        normalizer=None,
    )

    print(f"  Training clips        : {raw_ds.num_clips}")
    print(f"  Total training windows: {len(raw_ds):,}  (input dim: {raw_ds.input_dim})")

    n_val = max(1, int(len(raw_ds) * val_fraction))
    n_train = len(raw_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(raw_ds, [n_train, n_val], generator=generator)

    print("  Fitting normalizer on training windows...")
    train_data = raw_ds._all[train_subset.indices]
    normalizer = Normalizer().fit_array(train_data)
    print(f"  Normalizer fitted  (mean range: [{normalizer.mean.min():.3f}, {normalizer.mean.max():.3f}])")

    raw_ds.normalizer = normalizer
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, normalizer
