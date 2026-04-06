"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation pipeline for Auralytics.

Loads a trained checkpoint, runs inference on the full test set
(normal + anomalous clips), and computes:
  - AUC-ROC   (primary DCASE metric)
  - pAUC      (partial AUC over FPR 0-0.1)
  - F1        (at the threshold that maximises it)

Results are printed as a table and saved to results/.
Score distribution and ROC curve plots are also saved.

NOTE on threshold selection
─────────────────────────────────────────────────────────────────────────────
The val split in dataset.py contains only normal clips, so it cannot be used
for threshold selection or AUC computation. All evaluation here is done
against the full test set which contains both normal and anomalous clips.
This matches the DCASE 2020 Task 2 protocol exactly.

Usage:
    python -m src.evaluate --machine_type fan
    python -m src.evaluate --machine_type fan pump valve
    python -m src.evaluate --machine_type fan --show_plots
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from torch.utils.data import DataLoader

from src.dataset import DCASEDataset
from src.model import ConvAutoencoder
from src.utils import (
    get_device,
    load_checkpoint,
    plot_score_distribution,
    plot_reconstruction,
)

import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")


# ── Core inference ────────────────────────────────────────────────────────────

def collect_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over an entire DataLoader.

    Returns
    -------
    scores : (N,) float32 — reconstruction MSE per clip
    labels : (N,) int    — 0=normal, 1=anomalous
    """
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for specs, labels in loader:
            specs  = specs.to(device)
            scores = model.anomaly_score(specs)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    Compute partial AUC over FPR in [0, max_fpr], normalised to [0, 1].
    Matches the DCASE 2020 Task 2 pAUC definition.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # Keep only points within [0, max_fpr]
    mask = fpr <= max_fpr
    # Include the first point beyond the cutoff for smooth interpolation
    if not mask.all():
        cutoff = np.searchsorted(fpr, max_fpr, side="right")
        mask[cutoff] = True
    partial_fpr = fpr[mask]
    partial_tpr = tpr[mask]
    # Normalise by max_fpr so the result is in [0, 1]
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # np 2.0 renamed trapz
    return float(_trapz(partial_tpr, partial_fpr) / max_fpr)


def compute_f1_at_best_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """
    Sweep thresholds and return (best_f1, best_threshold).
    Uses the same threshold grid as the ROC curve for efficiency.
    """
    _, _, thresholds = roc_curve(labels, scores)
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_f1), float(best_thresh)


def evaluate_machine(
    machine_type: str,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path    = MODELS_DIR,
    results_dir: Path   = RESULTS_DIR,
    batch_size: int     = 32,
    show_plots: bool    = False,
) -> dict:
    """
    Full evaluation for one machine type.

    Returns a dict with keys: machine, auc, pauc, f1, threshold, n_normal, n_anomalous.
    """
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = models_dir / f"{machine_type}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Run: python -m src.train --machine_type {machine_type}"
        )
    model = ConvAutoencoder(base_ch=32).to(device)
    load_checkpoint(ckpt_path, model, device=device)

    # ── Load test set ─────────────────────────────────────────────────────────
    test_ds = DCASEDataset(processed_dir, machine_type, split="test")
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = test_ds.class_counts()
    print(f"\n  Test set: {counts['normal']} normal | {counts['anomalous']} anomalous")

    # ── Collect scores ────────────────────────────────────────────────────────
    scores, labels = collect_scores(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc             = float(roc_auc_score(labels, scores))
    pauc            = compute_pauc(labels, scores, max_fpr=0.1)
    f1, threshold   = compute_f1_at_best_threshold(labels, scores)

    # ── Plots ─────────────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    normal_scores    = scores[labels == 0]
    anomalous_scores = scores[labels == 1]

    plot_score_distribution(
        normal_scores, anomalous_scores, machine_type,
        threshold  = threshold,
        save_path  = results_dir / f"{machine_type}_score_dist.png",
    )

    _plot_roc(labels, scores, machine_type, auc,
              save_path=results_dir / f"{machine_type}_roc.png")

    if show_plots:
        plt.show()

    return {
        "machine":      machine_type,
        "auc":          auc,
        "pauc":         pauc,
        "f1":           f1,
        "threshold":    threshold,
        "n_normal":     int(counts["normal"]),
        "n_anomalous":  int(counts["anomalous"]),
    }


# ── ROC curve plot ────────────────────────────────────────────────────────────

def _plot_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    machine_type: str,
    auc: float,
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#0d0d0d")

    ax.plot(fpr, tpr, color="#f59e0b", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.axvline(0.1, color="#555", linestyle="--", linewidth=1, label="pAUC boundary (FPR=0.1)")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", color="#e8e8e8")
    ax.set_ylabel("True Positive Rate", color="#e8e8e8")
    ax.set_title(f"ROC Curve — {machine_type.upper()}", color="#e8e8e8", fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Auralytics — evaluate trained models")
    parser.add_argument(
        "--machine_type", nargs="+",
        default=["fan", "pump", "valve"],
        choices=["fan", "pump", "valve"],
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir",    type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir",   type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--show_plots",    action="store_true",
                        help="Display plots interactively (in addition to saving)")
    args = parser.parse_args()

    all_results = []
    for machine in args.machine_type:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {machine.upper()}")
        print(f"{'='*50}")
        try:
            result = evaluate_machine(
                machine_type  = machine,
                processed_dir = args.processed_dir,
                models_dir    = args.models_dir,
                results_dir   = args.results_dir,
                batch_size    = args.batch_size,
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

    print(f"\nPlots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
