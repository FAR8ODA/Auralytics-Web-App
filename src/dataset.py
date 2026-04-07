"""
src/dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Datasets for Auralytics — frame-level MLP autoencoder.

Two dataset classes:

  FrameDataset   — used for TRAINING.
                   Slices every clip into overlapping windows of N_FRAMES
                   consecutive mel columns. Each window is one training sample.
                   Labels are not exposed (unsupervised — train on normal only).

  ClipDataset    — used for EVALUATION.
                   Returns the full spectrogram for each clip so the evaluator
                   can slice it into windows, score each window, and aggregate
                   to a single clip-level anomaly score.
                   Exposes labels (0=normal, 1=anomalous) for AUC computation.

Window geometry (defaults)
  N_MELS    = 128    mel bins  (set by preprocess.py)
  N_FRAMES  = 5      consecutive time frames per window
  HOP       = 1      step between windows (dense — every frame produces a window)
  INPUT_DIM = 128 * 5 = 640   flattened input to the MLP

Label convention
  0 → normal
  1 → anomalous
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
N_FRAMES  = 5    # frames per window
HOP       = 1    # step between windows during training
N_MELS    = 128  # must match preprocess.py
INPUT_DIM = N_MELS * N_FRAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_from_filename(stem: str) -> int:
    return 1 if stem.startswith("anomaly") else 0


def extract_windows(spec: np.ndarray, n_frames: int = N_FRAMES, hop: int = HOP) -> np.ndarray:
    """
    Slice a (N_MELS, T) spectrogram into overlapping windows.

    Returns: (n_windows, N_MELS * n_frames) float32 array.
    Each row is one flattened window — the input to the MLP.
    """
    n_mels, T = spec.shape
    starts = range(0, T - n_frames + 1, hop)
    windows = np.stack(
        [spec[:, t : t + n_frames].flatten() for t in starts],
        axis=0,
    )
    return windows.astype(np.float32)


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Global mean/std normalizer fitted on training windows.

    Fit once on the training FrameDataset, then apply to val and test.
    Fitting on test data would be data leakage.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, dataset: "FrameDataset") -> "Normalizer":
        """Compute mean and std over all training windows."""
        all_windows = []
        for i in range(len(dataset)):
            w, _ = dataset[i]
            all_windows.append(w.numpy())
        data = np.stack(all_windows, axis=0)
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std  = (data.std(axis=0) + 1e-8).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() before transform()"
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        data = np.load(path)
        norm = cls()
        norm.mean = data["mean"]
        norm.std  = data["std"]
        return norm


# ── FrameDataset (training) ───────────────────────────────────────────────────

class FrameDataset(Dataset):
    """
    One training sample = one flattened window of N_FRAMES mel columns.

    Built from the train split (normal clips only).
    The normalizer is fitted externally and passed in after fitting.
    """

    def __init__(
        self,
        processed_dir: Path,
        machine_type:  str,
        split:         str = "train",
        n_frames:      int = N_FRAMES,
        hop:           int = HOP,
        normalizer:    Optional[Normalizer] = None,
    ):
        split_dir = Path(processed_dir) / machine_type / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found at {split_dir}. "
                f"Run: python -m src.preprocess --machine_types {machine_type}"
            )

        self.n_frames   = n_frames
        self.normalizer = normalizer
        self._windows: list[np.ndarray] = []

        files = sorted(split_dir.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No .npy files in {split_dir}")

        for f in files:
            spec    = np.load(f)          # (N_MELS, T)
            windows = extract_windows(spec, n_frames, hop)
            self._windows.append(windows)

        self._all = np.concatenate(self._windows, axis=0)  # (total_windows, INPUT_DIM)

    def __len__(self) -> int:
        return len(self._all)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        w = self._all[idx]
        if self.normalizer is not None:
            w = self.normalizer.transform(w)
        return torch.from_numpy(w), 0   # label unused during training

    @property
    def input_dim(self) -> int:
        return self._all.shape[1]


# ── ClipDataset (evaluation) ──────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    One sample = one full clip spectrogram + its label.

    Used exclusively for evaluation. The evaluator slices each clip into
    windows, scores each window, and aggregates to a clip-level score.
    """

    def __init__(
        self,
        processed_dir: Path,
        machine_type:  str,
        split:         str = "test",
    ):
        split_dir = Path(processed_dir) / machine_type / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Processed data not found at {split_dir}")

        self.files  = sorted(split_dir.glob("*.npy"))
        self.labels = [_label_from_filename(f.stem) for f in self.files]

        if not self.files:
            raise RuntimeError(f"No .npy files in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        spec  = np.load(self.files[idx]).astype(np.float32)   # (N_MELS, T)
        label = self.labels[idx]
        return spec, label

    def class_counts(self) -> dict:
        n_anom = sum(self.labels)
        return {"normal": len(self.labels) - n_anom, "anomalous": n_anom}


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_train_val_loaders(
    processed_dir: Path,
    machine_type:  str,
    batch_size:    int   = 512,
    num_workers:   int   = 0,
    val_fraction:  float = 0.1,
    seed:          int   = 42,
    n_frames:      int   = N_FRAMES,
    hop:           int   = HOP,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Normalizer]:
    """
    Build train and val DataLoaders for one machine type.

    Fits the global normalizer on the training windows and applies it to both.
    Returns (train_loader, val_loader, normalizer).
    The normalizer must be saved and reused at evaluation time.
    """
    from torch.utils.data import DataLoader, random_split

    # Build raw (unnormalised) dataset first so we can fit the normalizer
    raw_ds = FrameDataset(processed_dir, machine_type, split="train",
                          n_frames=n_frames, hop=hop, normalizer=None)

    print(f"  Total training windows : {len(raw_ds):,}  (input dim: {raw_ds.input_dim})")

    # Train / val split
    n_val   = max(1, int(len(raw_ds) * val_fraction))
    n_train = len(raw_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_raw, val_raw = random_split(raw_ds, [n_train, n_val], generator=generator)

    # Fit normalizer on training windows only
    norm = Normalizer()
    print("  Fitting normalizer on training windows...")
    # Efficient: use the underlying numpy array directly
    train_indices = train_raw.indices
    train_data    = raw_ds._all[train_indices]
    norm.mean     = train_data.mean(axis=0).astype(np.float32)
    norm.std      = (train_data.std(axis=0) + 1e-8).astype(np.float32)
    print(f"  Normalizer fitted  (mean range: [{norm.mean.min():.3f}, {norm.mean.max():.3f}])")

    # Reuse the already-built dataset instead of reloading every file again.
    # This avoids a second expensive pass over >1M windows and is much more
    # stable on Colab than rebuilding FrameDataset after fitting the normalizer.
    raw_ds.normalizer = norm

    train_loader = DataLoader(train_raw, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_raw,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm
