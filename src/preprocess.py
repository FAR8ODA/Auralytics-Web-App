"""Audio preprocessing pipeline for Auralytics.

Converts DCASE wav clips into raw log-mel dB spectrograms saved as .npy files.
Important: this final pipeline does not standardize each clip independently.
Per-clip normalization erased useful amplitude differences; normalization is
instead learned globally from training windows in src/train.py.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
DURATION = 10.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def load_audio(path: Path) -> np.ndarray:
    """Load a wav file as mono 16 kHz audio with fixed 10 second length."""
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=DURATION)
    if len(waveform) < N_SAMPLES:
        waveform = np.pad(waveform, (0, N_SAMPLES - len(waveform)))
    return waveform[:N_SAMPLES]


def compute_log_mel(waveform: np.ndarray) -> np.ndarray:
    """Convert waveform to log-mel dB spectrogram, shape (128, T)."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def preprocess_split(raw_split_dir: Path, out_split_dir: Path) -> dict:
    """Process all wav files in one split and cache them as npy spectrograms."""
    out_split_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(raw_split_dir.glob("*.wav"))

    if not wav_files:
        print(f"  [warn] no .wav files found in {raw_split_dir}")
        return {"processed": 0, "skipped": 0}

    processed, skipped = 0, 0
    for wav_path in tqdm(wav_files, desc=f"  {raw_split_dir.parent.name}/{raw_split_dir.name}", leave=False):
        out_path = out_split_dir / f"{wav_path.stem}.npy"
        if out_path.exists():
            skipped += 1
            continue
        try:
            spec = compute_log_mel(load_audio(wav_path))
            np.save(out_path, spec)
            processed += 1
        except Exception as exc:
            print(f"  [error] {wav_path.name}: {exc}")

    return {"processed": processed, "skipped": skipped}


def preprocess_machine(machine_type: str, data_dir: Path, out_dir: Path) -> None:
    """Process train and test splits for one machine type."""
    print(f"\n-- {machine_type.upper()} --")
    for split in ["train", "test"]:
        raw_split = data_dir / machine_type / split
        out_split = out_dir / machine_type / split

        if not raw_split.exists():
            print(f"  [skip] {raw_split} not found - download data first")
            continue

        stats = preprocess_split(raw_split, out_split)
        print(f"  {split:5s} -> processed: {stats['processed']:4d} | skipped: {stats['skipped']:4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auralytics audio preprocessor")
    parser.add_argument(
        "--machine_types",
        nargs="+",
        default=["fan", "pump", "valve"],
        help="Machine types to process (default: fan pump valve)",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Root directory of raw DCASE audio")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"), help="Output directory for npy files")
    args = parser.parse_args()

    print("Auralytics - Preprocessing Pipeline")
    print(f"  source : {args.data_dir}")
    print(f"  output : {args.out_dir}")
    print(f"  types  : {', '.join(args.machine_types)}")
    print(f"  config : {N_MELS} mels | FFT {N_FFT} | hop {HOP_LENGTH} | {SAMPLE_RATE} Hz")

    for machine_type in args.machine_types:
        preprocess_machine(machine_type, args.data_dir, args.out_dir)

    print("\nDone. Train with python -m src.train after preprocessing.")


if __name__ == "__main__":
    main()
