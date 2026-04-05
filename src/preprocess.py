"""
src/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Audio preprocessing pipeline for Auralytics.

Converts raw .wav clips into log-mel spectrograms and saves them as .npy
files under data/processed/{machine_type}/{split}/.

Usage:
    python src/preprocess.py --machine_types fan pump valve
    python src/preprocess.py --machine_types fan --data_dir data/raw --out_dir data/processed
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# ── Spectrogram config ────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # DCASE clips are all 16 kHz
N_MELS        = 128      # mel filterbank size
N_FFT         = 1024     # FFT window length
HOP_LENGTH    = 512      # hop between frames
DURATION      = 10.0     # clip length in seconds (pad/truncate to this)
N_SAMPLES     = int(SAMPLE_RATE * DURATION)


def load_audio(path: Path) -> np.ndarray:
    """Load a wav file, resample to SAMPLE_RATE, mono, fixed length."""
    waveform, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=DURATION)
    # Pad if shorter than expected (rare but possible)
    if len(waveform) < N_SAMPLES:
        waveform = np.pad(waveform, (0, N_SAMPLES - len(waveform)))
    return waveform[:N_SAMPLES]


def compute_log_mel(waveform: np.ndarray) -> np.ndarray:
    """Convert waveform to log-mel spectrogram, shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize(spec: np.ndarray) -> np.ndarray:
    """Normalize spectrogram to zero mean, unit variance (per clip)."""
    mean = spec.mean()
    std  = spec.std() + 1e-8   # avoid divide-by-zero on silence
    return (spec - mean) / std


def preprocess_split(
    raw_split_dir: Path,
    out_split_dir: Path,
) -> dict:
    """
    Process all .wav files in raw_split_dir and save .npy spectrograms
    to out_split_dir.  Returns a small stats dict.
    """
    out_split_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(raw_split_dir.glob("*.wav"))

    if not wav_files:
        print(f"  [warn] no .wav files found in {raw_split_dir}")
        return {"processed": 0, "skipped": 0}

    processed, skipped = 0, 0
    for wav_path in tqdm(wav_files, desc=f"  {raw_split_dir.parent.name}/{raw_split_dir.name}", leave=False):
        out_path = out_split_dir / (wav_path.stem + ".npy")
        if out_path.exists():
            skipped += 1
            continue
        try:
            waveform = load_audio(wav_path)
            spec     = compute_log_mel(waveform)
            spec     = normalize(spec)
            np.save(out_path, spec)
            processed += 1
        except Exception as e:
            print(f"  [error] {wav_path.name}: {e}")

    return {"processed": processed, "skipped": skipped}


def preprocess_machine(
    machine_type: str,
    data_dir: Path,
    out_dir: Path,
) -> None:
    """Process train and test splits for a single machine type."""
    print(f"\n── {machine_type.upper()} ──────────────────────────────────")
    for split in ["train", "test"]:
        raw_split = data_dir / machine_type / split
        out_split = out_dir  / machine_type / split

        if not raw_split.exists():
            print(f"  [skip] {raw_split} not found — download data first")
            continue

        stats = preprocess_split(raw_split, out_split)
        print(f"  {split:5s} → processed: {stats['processed']:4d}  |  skipped (cached): {stats['skipped']:4d}")


def main():
    parser = argparse.ArgumentParser(description="Auralytics audio preprocessor")
    parser.add_argument(
        "--machine_types", nargs="+", default=["fan", "pump", "valve"],
        help="Machine types to process (default: fan pump valve)"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/raw"),
        help="Root directory of raw DCASE audio"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("data/processed"),
        help="Root directory for output spectrograms"
    )
    args = parser.parse_args()

    print("Auralytics — Preprocessing Pipeline")
    print(f"  source : {args.data_dir}")
    print(f"  output : {args.out_dir}")
    print(f"  types  : {', '.join(args.machine_types)}")
    print(f"  config : {N_MELS} mels | FFT {N_FFT} | hop {HOP_LENGTH} | {SAMPLE_RATE} Hz")

    for machine_type in args.machine_types:
        preprocess_machine(machine_type, args.data_dir, args.out_dir)

    print("\nDone. Run notebooks/01_eda.ipynb to inspect the spectrograms.")


if __name__ == "__main__":
    main()
