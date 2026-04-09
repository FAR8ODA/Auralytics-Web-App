"""Evaluation pipeline for the Auralytics frame-level MLP autoencoder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from src.dataset import ClipDataset, INPUT_DIM, N_FRAMES, Normalizer, available_machine_ids, extract_windows
from src.model import MLPAutoencoder
from src.utils import get_device, load_checkpoint, plot_score_distribution

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
AggMode = Literal["mean", "p95"]


def run_tag(machine_type: str, machine_id: Optional[str] = None) -> str:
    return f"{machine_type}_{machine_id}" if machine_id else machine_type


def scope_label(machine_type: str, machine_id: Optional[str] = None) -> str:
    return f"{machine_type}/{machine_id}" if machine_id else machine_type


def save_results_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["machine", "auc", "pauc", "f1", "threshold", "n_normal", "n_anomalous"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def score_clip(
    spec: np.ndarray,
    model: torch.nn.Module,
    normalizer: Normalizer,
    device: torch.device,
    n_frames: int = N_FRAMES,
    agg: AggMode = "mean",
) -> float:
    windows = extract_windows(spec, n_frames=n_frames, hop=1)
    windows = normalizer.transform(windows)
    tensor = torch.from_numpy(windows).to(device)
    with torch.no_grad():
        errors = model.anomaly_score_windows(tensor).cpu().numpy()
    if agg == "p95":
        return float(np.percentile(errors, 95))
    return float(errors.mean())


def collect_scores(
    model: torch.nn.Module,
    dataset: ClipDataset,
    normalizer: Normalizer,
    device: torch.device,
    n_frames: int = N_FRAMES,
    agg: AggMode = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    model.eval()
    for idx in range(len(dataset)):
        spec, label = dataset[idx]
        scores.append(score_clip(spec, model, normalizer, device, n_frames=n_frames, agg=agg))
        labels.append(label)
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=int)


def compute_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= max_fpr
    if not mask.all():
        cutoff = np.searchsorted(fpr, max_fpr, side="right")
        mask[min(cutoff, len(mask) - 1)] = True
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tpr[mask], fpr[mask]) / max_fpr)


def compute_f1_at_best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    _, _, thresholds = roc_curve(labels, scores)
    best_f1 = 0.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_f1, best_threshold


def evaluate_machine(
    machine_type: str,
    machine_id: Optional[str] = None,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    n_frames: int = N_FRAMES,
    agg: AggMode = "mean",
    show_plots: bool = False,
) -> dict:
    device = get_device()
    tag = run_tag(machine_type, machine_id)

    ckpt_path = models_dir / f"{tag}_mlp_best.pth"
    norm_path = models_dir / f"{tag}_normalizer.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run: python -m src.train --machine_type {machine_type} --machine_id {machine_id}"
        )
    if not norm_path.exists():
        raise FileNotFoundError(f"No normalizer at {norm_path}. Re-run training for {scope_label(machine_type, machine_id)}")

    model = MLPAutoencoder(input_dim=INPUT_DIM, bottleneck=8).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    normalizer = Normalizer.load(norm_path)

    test_ds = ClipDataset(processed_dir, machine_type, split="test", machine_id=machine_id)
    counts = test_ds.class_counts()
    print(f"\n  Test set: {counts['normal']} normal | {counts['anomalous']} anomalous")
    print(f"  Scoring clips (agg={agg})...")

    scores, labels = collect_scores(model, test_ds, normalizer, device, n_frames=n_frames, agg=agg)
    auc = float(roc_auc_score(labels, scores))
    pauc = compute_pauc(labels, scores)
    f1, threshold = compute_f1_at_best_threshold(labels, scores)

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(
        scores[labels == 0],
        scores[labels == 1],
        tag,
        threshold=threshold,
        save_path=results_dir / f"{tag}_mlp_score_dist.png",
    )
    _plot_roc(labels, scores, tag, auc, save_path=results_dir / f"{tag}_mlp_roc.png")
    if show_plots:
        plt.show()

    return {
        "machine": scope_label(machine_type, machine_id),
        "auc": auc,
        "pauc": pauc,
        "f1": f1,
        "threshold": threshold,
        "n_normal": int(counts["normal"]),
        "n_anomalous": int(counts["anomalous"]),
    }


def _plot_roc(labels: np.ndarray, scores: np.ndarray, tag: str, auc: float, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#0d0d0d")
    ax.plot(fpr, tpr, color="#f59e0b", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.axvline(0.1, color="#555", linestyle="--", linewidth=1, label="pAUC boundary")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", color="#e8e8e8")
    ax.set_ylabel("True Positive Rate", color="#e8e8e8")
    ax.set_title(f"ROC Curve - {tag.upper()} (MLP AE)", color="#e8e8e8", fontweight="bold")
    ax.tick_params(colors="#888")
    ax.legend(facecolor="#1a1a1a", labelcolor="#e8e8e8")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def print_results_table(results: list[dict]) -> None:
    header = f"\n{'Machine':<14} {'AUC':>7} {'pAUC':>7} {'F1':>7} {'Threshold':>12}"
    print(header)
    print("-" * len(header))
    for result in results:
        flag = "  <-- below target" if result["auc"] < 0.75 else ""
        print(
            f"{result['machine']:<14} {result['auc']:>7.3f} {result['pauc']:>7.3f} "
            f"{result['f1']:>7.3f} {result['threshold']:>12.5f}{flag}"
        )
    aucs = [result["auc"] for result in results]
    print(f"\n  Mean AUC : {np.mean(aucs):.3f}")
    print(f"  Min  AUC : {np.min(aucs):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auralytics MLP AE - evaluate")
    parser.add_argument("--machine_type", nargs="+", default=["fan", "pump", "valve"], choices=["fan", "pump", "valve"])
    parser.add_argument("--machine_id", default=None, help="Optional machine ID such as id_00")
    parser.add_argument("--all_ids", action="store_true", help="Evaluate every available ID for one machine type")
    parser.add_argument("--list_ids", action="store_true", help="Print available machine IDs and exit")
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--agg", default="mean", choices=["mean", "p95"], help="Window score aggregation method")
    parser.add_argument("--save_csv", type=Path, default=None, help="Optional path to save a CSV summary of the results")
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()

    if args.machine_id and args.all_ids:
        raise SystemExit("Use either --machine_id or --all_ids, not both")
    if args.list_ids:
        if len(args.machine_type) != 1:
            raise SystemExit("--list_ids expects exactly one machine type")
        for machine_id in available_machine_ids(args.processed_dir, args.machine_type[0], split="test"):
            print(machine_id)
        return
    if args.all_ids and len(args.machine_type) != 1:
        raise SystemExit("--all_ids can only be used with one machine type at a time")

    results: list[dict] = []
    if args.all_ids:
        machine_type = args.machine_type[0]
        targets = [(machine_type, machine_id) for machine_id in available_machine_ids(args.processed_dir, machine_type, split="test")]
    elif args.machine_id:
        machine_type = args.machine_type[0]
        ids = available_machine_ids(args.processed_dir, machine_type, split="test")
        if args.machine_id not in ids:
            available = ", ".join(ids) if ids else "none"
            raise SystemExit(f"Unknown machine_id {args.machine_id!r} for {machine_type}. Available IDs: {available}")
        targets = [(machine_type, args.machine_id)]
    else:
        targets = [(machine_type, None) for machine_type in args.machine_type]

    for machine_type, machine_id in targets:
        print(f"\n{'=' * 50}")
        suffix = f" ({machine_id})" if machine_id else ""
        print(f"  Evaluating: {machine_type.upper()}{suffix}  (MLP AE)")
        print(f"{'=' * 50}")
        try:
            result = evaluate_machine(
                machine_type=machine_type,
                machine_id=machine_id,
                processed_dir=args.processed_dir,
                models_dir=args.models_dir,
                results_dir=args.results_dir,
                agg=args.agg,
                show_plots=args.show_plots,
            )
            results.append(result)
            print(f"  AUC  : {result['auc']:.3f}")
            print(f"  pAUC : {result['pauc']:.3f}")
            print(f"  F1   : {result['f1']:.3f}")
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")

    if len(results) > 1:
        print_results_table(results)
    if args.save_csv and results:
        save_results_csv(results, args.save_csv)
        print(f"\nSaved results CSV to {args.save_csv}")
    elif args.all_ids and results:
        csv_path = args.results_dir / f"{args.machine_type[0]}_all_ids_summary.csv"
        save_results_csv(results, csv_path)
        print(f"\nSaved results CSV to {csv_path}")
    if results:
        print(f"\nPlots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
