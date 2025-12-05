"""
Visualize robustness summary metrics produced by mnist_mlp_robust_training.py.

Usage example:
    python examples/vision/plot_mnist_mlp_robust_summary.py \
        --summary ./results/mnist_mlp_robust_mlp/summary.json \
        --out-dir ./results/mnist_mlp_robust_mlp/viz \
        --show
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> List[Dict]:
    with path.open() as f:
        return json.load(f)


def extract_cert_matrix(
    summary: List[Dict],
    stage: str,
) -> Tuple[np.ndarray, List[float], List[float]]:
    train_eps = []
    eval_eps = set()
    for rec in summary:
        if stage not in rec:
            continue
        train_eps.append(rec["epsilon"])
        for key in rec[stage]["eval"].keys():
            if key.startswith("cert_acc_eps_"):
                eval_eps.add(float(key.split("_")[-1]))
    train_eps = sorted(set(train_eps))
    eval_eps = sorted(eval_eps)
    matrix = np.full((len(train_eps), len(eval_eps)), np.nan)
    for i, eps in enumerate(train_eps):
        rec = next((r for r in summary if r["epsilon"] == eps and stage in r), None)
        if not rec:
            continue
        eval_data = rec[stage]["eval"]
        for j, e in enumerate(eval_eps):
            key = f"cert_acc_eps_{e}"
            if key in eval_data:
                matrix[i, j] = eval_data[key]
    return matrix, train_eps, eval_eps


def plot_heatmap(
    data: np.ndarray,
    x_labels: List[float],
    y_labels: List[float],
    title: str,
    out_path: Path,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    img = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([f"{x:.3f}" for x in x_labels], rotation=45)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([f"{y:.3f}" for y in y_labels])
    ax.set_xlabel("Evaluation epsilon")
    ax.set_ylabel("Training epsilon")
    ax.set_title(title)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Certified ratio")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_cert_curves(
    summary: List[Dict],
    stages: List[str],
    out_dir: Path,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    # Individual plots
    for stage in stages:
        fig, ax = plt.subplots(figsize=(6, 4))
        for rec in summary:
            if stage not in rec:
                continue
            eps_train = rec["epsilon"]
            eval_pairs = sorted(
                (float(k.split("_")[-1]), v)
                for k, v in rec[stage]["eval"].items()
                if k.startswith("cert_acc_eps_")
            )
            if not eval_pairs:
                continue
            x, y = zip(*eval_pairs)
            ax.plot(x, y, marker="o", label=f"eps={eps_train:.3f}")
        ax.set_xlabel("Evaluation epsilon")
        ax.set_ylabel("Certified ratio")
        ax.set_title(f"Certified curves ({stage})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig((out_dir / f"curves_{stage}").with_suffix(f".{fmt}"), dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

    # Pairwise plots
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            stage_a, stage_b = stages[i], stages[j]
            for rec in summary:
                if stage_a not in rec or stage_b not in rec:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                title_eps = rec["epsilon"]
                for idx, stage in enumerate([stage_a, stage_b]):
                    eval_pairs = sorted(
                        (float(k.split("_")[-1]), v)
                        for k, v in rec[stage]["eval"].items()
                        if k.startswith("cert_acc_eps_")
                    )
                    if not eval_pairs:
                        continue
                    x, y = zip(*eval_pairs)
                    color = palette[idx % len(palette)]
                    marker = "o" if idx == 0 else "s"
                    ax.plot(
                        x,
                        y,
                        marker=marker,
                        linestyle="-" if idx == 0 else "--",
                        color=color,
                        alpha=0.9,
                        label=f"{stage} ({title_eps:.3f})",
                    )
                ax.set_xlabel("Evaluation epsilon")
                ax.set_ylabel("Certified ratio")
                ax.set_title(f"{stage_a} vs {stage_b} @ train eps {title_eps:.3f}")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fname = f"curves_{stage_a}_vs_{stage_b}_eps{title_eps:.3f}.png"
                fig.savefig((out_dir / fname).with_suffix(f".{fmt}"), dpi=dpi)
                if show:
                    plt.show()
                plt.close(fig)


def plot_accuracy_bars(
    summary: List[Dict],
    stage_a: str,
    stage_b: str,
    save_path: Path,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    eps_list = sorted({rec["epsilon"] for rec in summary})
    width = 0.35
    x = np.arange(len(eps_list))
    fig, ax = plt.subplots(figsize=(6, 4))
    def acc_for_stage(stage: str) -> List[float]:
        vals = []
        for eps in eps_list:
            rec = next((r for r in summary if r["epsilon"] == eps and stage in r), None)
            vals.append(rec[stage].get("test_acc", np.nan) if rec else np.nan)
        return vals
    acc_a = acc_for_stage(stage_a)
    acc_b = acc_for_stage(stage_b)
    ax.bar(x - width/2, acc_a, width, label=stage_a)
    ax.bar(x + width/2, acc_b, width, label=stage_b)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eps:.2f}" for eps in eps_list])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training epsilon")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Natural accuracy comparison")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustness summary visualizations.")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json produced by training script.")
    parser.add_argument("--out-dir", type=str, default="./results/mnist_mlp_robust_mlp/viz", help="Directory to save plots.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="File format for saved figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving.")
    parser.add_argument("--stages", type=str, nargs="+", default=["finetuned", "direct"], help="Which stages to visualize.")
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    out_dir = Path(args.out_dir)
    for stage in args.stages:
        matrix, train_eps, eval_eps = extract_cert_matrix(summary, stage)
        if train_eps and eval_eps:
            plot_heatmap(matrix, eval_eps, train_eps, f"Certified ratios ({stage})", out_dir / f"heatmap_{stage}", args.format, args.dpi, args.show)
    if args.stages:
        plot_cert_curves(summary, args.stages, out_dir, args.format, args.dpi, args.show)
    if len(args.stages) >= 2:
        plot_accuracy_bars(summary, args.stages[0], args.stages[1], out_dir / "test_acc_compare", args.format, args.dpi, args.show)


if __name__ == "__main__":
    main()

