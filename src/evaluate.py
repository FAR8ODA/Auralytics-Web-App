"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation pipeline for the Auralytics MLP frame-level autoencoder.

Clip-level scoring strategy
─────────────────────────────────────────────────────────────────────────────
1. Slice the clip spectrogram into overlapping windows (same N_FRAMES, hop=1)
2. Apply the saved normalizer (fitted on training data — never on test)
3. Run each window through the MLP autoencoder
4. Collect per-window reconstruction errors
5. Aggregate to one clip score:
     default  → mean error across all windows
     optional → 95th-percentile (catches localised anomalies better)
6. Use clip scores for AUC-ROC, pAUC, F1 computation

Usage:
    python -m src.evaluate --machine_type fan
    python -m src.evaluate --machine_type fan pump valve
    python -m src.evaluate --machine_type fan --aggregation p95
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from torch.utils.data import DataLoader

from src.dataset import ClipDataset, Normalizer, extract_windows, N_FRAMES
from src.model import MLPAutoencoder, INPUT_DIM
from src.utils import (
    get_device,
    load_checkpoint,
    plot_score_distribution,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")

AggMode = Literal["mean", "p95"]


# ── Clip-level scoring ────────────────────────────────────────────────────────

def score_clip(
    spec:       np.ndarray,
    model:      torch.nn.Module,
    normalizer: Normalizer,
    device:     torch.device,
    n_frames:   int     = N_FRAMES,
    agg:        AggMode = "mean",
) -> float:
    """
    Score a single clip spectrogram → one anomaly score.

    Args:
        spec       : (N_MELS, T) spectrogram from ClipDataset
        model      : trained MLPAutoencoder
        normalizer : fitted on training data — applied here to avoid leakage
        agg        : 'mean' or 'p95' window error aggregation
    """
    windows = extract_windows(spec, n_frames=n_frames, hop=1)   # (W, INPUT_DIM)
    windows = normalizer.transform(windows)                      # global normalisation
    tensor  = torch.from_numpy(windows).to(device)              # (W, INPUT_DIM)

    with torch.no_grad():
        errors = model.anomaly_score_windows(tensor).cpu().numpy()  # (W,)

    if agg == "p95":
        return float(np.percentile(errors, 95))
    return float(errors.mean())


def collect_scores(
    model:      torch.nn.Module,
    dataset:    ClipDataset,
    normalizer: Normalizer,
    device:     torch.device,
    n_frames:   int     = N_FRAMES,
    agg:        AggMode = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over an entire ClipDataset.

    Returns
    -------
    scores : (N,) float32 — one anomaly score per clip
    labels : (N,) int    — 0=normal, 1=anomalous
    """
    all_scores, all_labels = [], []
    model.eval()
    for i in range(len(dataset)):
        spec, label = dataset[i]
        score = score_clip(spec, model, normalizer, device, n_frames, agg)
        all_scores.append(score)
        all_labels.append(label)
    return np.array(all_scores, dtype=np.float32), np.array(all_labels, dtype=int)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= max_fpr
    if not mask.all():
        cutoff = np.searchsorted(fpr, max_fpr, side="right")
        mask[min(cutoff, len(mask) - 1)] = True
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(tpr[mask], fpr[mask]) / max_fpr)


def compute_f1_at_best_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    _, _, thresholds = roc_curve(labels, scores)
    best_f1, best_thresh = 0.0, float(thresholds[0])
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return best_f1, best_thresh


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_machine(
    machine_type:  str,
    processed_dir: Path    = PROCESSED_DIR,
    models_dir:    Path    = MODELS_DIR,
    results_dir:   Path    = RESULTS_DIR,
    n_frames:      int     = N_FRAMES,
    agg:           AggMode = "mean",
    show_plots:    bool    = False,
) -> dict:
    """
    Full evaluation for one machine type.

    Returns dict with: machine, auc, pauc, f1, threshold, n_normal, n_anomalous.
    """
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = models_dir / f"{machine_type}_mlp_best.pth"
    norm_path = models_dir / f"{machine_type}_normalizer.npz"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. "
            f"Run: python -m src.train --machine_type {machine_type}"
        )
    if not norm_path.exists():
        raise FileNotFoundError(
            f"No normalizer at {norm_path}. "
            f"Re-run training — normalizer is saved automatically."
        )

    model      = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=128).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    normalizer = Normalizer.load(norm_path)

    # ── Load test set ─────────────────────────────────────────────────────────
    test_ds = ClipDataset(processed_dir, machine_type, split="test")
    counts  = test_ds.class_counts()
    print(f"\n  Test set: {counts['normal']} normal | {counts['anomalous']} anomalous")
    print(f"  Scoring clips (agg={agg})...")

    # ── Collect clip-level scores ─────────────────────────────────────────────
    scores, labels = collect_scores(model, test_ds, normalizer, device, n_frames, agg)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc            = float(roc_auc_score(labels, scores))
    pauc           = compute_pauc(labels, scores)
    f1, threshold  = compute_f1_at_best_threshold(labels, scores)

    # ── Plots ─────────────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_score_distribution(
        scores[labels == 0], scores[labels == 1], machine_type,
        threshold = threshold,
        save_path = results_dir / f"{machine_type}_mlp_score_dist.png",
    )
    _plot_roc(
        labels, scores, machine_type, auc,
        save_path=results_dir / f"{machine_type}_mlp_roc.png",
    )

    if show_plots:
        plt.show()

    return {
        "machine":     machine_type,
        "auc":         auc,
        "pauc":        pauc,
        "f1":          f1,
        "threshold":   threshold,
        "n_normal":    int(counts["normal"]),
        "n_anomalous": int(counts["anomalous"]),
    }


# ── ROC plot ──────────────────────────────────────────────────────────────────

def _plot_roc(
    labels: np.ndarray, scores: np.ndarray,
    machine_type: str, auc: float, save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#1a1a1a");  fig.patch.set_facecolor("#0d0d0d")
    ax.plot(fpr, tpr, color="#f59e0b", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.axvline(0.1, color="#555", linestyle="--", linewidth=1, label="pAUC boundary")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", color="#e8e8e8")
    ax.set_ylabel("True Positive Rate", color="#e8e8e8")
    ax.set_title(f"ROC Curve — {machine_type.upper()} (MLP AE)", color="#e8e8e8", fontweight="bold")
    ax.tick_params(colors="#888")
    ax.legend(facecolor="#1a1a1a", labelcolor="#e8e8e8")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    header = f"\n{'Machine':<10} {'AUC':>7} {'pAUC':>7} {'F1':>7} {'Threshold':>12}"
    print(header)
    print("─" * len(header))
    for r in results:
        flag = "  <-- below target" if r["auc"] < 0.75 else ""
        print(
            f"{r['machine']:<10} {r['auc']:>7.3f} {r['pauc']:>7.3f} "
            f"{r['f1']:>7.3f} {r['threshold']:>12.5f}{flag}"
        )
    aucs = [r["auc"] for r in results]
    print(f"\n  Mean AUC : {np.mean(aucs):.3f}")
    print(f"  Min  AUC : {np.min(aucs):.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auralytics MLP AE — evaluate")
    parser.add_argument("--machine_type", nargs="+", default=["fan", "pump", "valve"],
                        choices=["fan", "pump", "valve"])
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir",    type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir",   type=Path, default=RESULTS_DIR)
    parser.add_argument("--agg",           default="mean", choices=["mean", "p95"],
                        help="Window score aggregation method")
    parser.add_argument("--show_plots",    action="store_true")
    args = parser.parse_args()

    all_results = []
    for machine in args.machine_type:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {machine.upper()}  (MLP AE)")
        print(f"{'='*50}")
        try:
            result = evaluate_machine(
                machine_type  = machine,
                processed_dir = args.processed_dir,
                models_dir    = args.models_dir,
                results_dir   = args.results_dir,
                agg           = args.agg,
                show_plots    = args.show_plots,
            )
            all_results.append(result)
            print(f"  AUC  : {result['auc']:.3f}")
            print(f"  pAUC : {result['pauc']:.3f}")
            print(f"  F1   : {result['f1']:.3f}")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    if len(all_results) > 1:
        print_results_table(all_results)

    if all_results:
        print(f"\nPlots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
