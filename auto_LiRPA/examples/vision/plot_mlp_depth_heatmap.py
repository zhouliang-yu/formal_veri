"""
Visualize robustness metrics as heatmaps over (epsilon vs depth/dropout/width).

Example usage (depth on vertical axis, single dropout per depth):
    python plot_mlp_depth_heatmap.py --bounds-dir ./results/mnist_mlp_depth \
        --save-path ./results/.../depth_heatmaps.png

For dropout sweep at fixed depth:
    python plot_mlp_depth_heatmap.py --y-axis dropout --depth-filter 2 \
        --bounds-dir ./results/mnist_mlp_depth --save-path ./results/.../dropout_heatmaps.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ComboKey = Tuple[int, float, int]


def load_bounds(bounds_dir: Path, method: str) -> Tuple[Dict[ComboKey, Dict[float, Dict[str, float]]], List[float]]:
    """Load bound summaries grouped by (depth, dropout, hidden_dim)."""
    combos: Dict[ComboKey, Dict[float, Dict[str, float]]] = {}
    eps_values = set()
    for bounds_file in sorted(bounds_dir.glob("bounds_depth*.json")):
        stem = bounds_file.stem  # e.g., bounds_depth3 or bounds_depth3_drop0p10
        try:
            after_depth = stem.split("depth", 1)[1]
        except IndexError:
            print(f"Skipping {bounds_file} (cannot parse depth).")
            continue
        depth_str = after_depth.split("_", 1)[0]
        try:
            depth_from_name = int(depth_str)
        except ValueError:
            print(f"Skipping {bounds_file} (cannot parse depth).")
            continue

        with bounds_file.open() as f:
            entries = json.load(f)
        filtered = [entry for entry in entries if entry.get("method") == method]
        if not filtered:
            print(f"No entries for method {method} in {bounds_file}; skipping.")
            continue

        for entry in filtered:
            eps = float(entry["eps"])
            entry_depth = int(entry.get("depth", depth_from_name))
            dropout = float(entry.get("dropout", 0.0))
            hidden_dim = int(entry.get("hidden_dim", 0))
            key = (entry_depth, dropout, hidden_dim)
            combos.setdefault(key, {})[eps] = entry
            eps_values.add(eps)

    if not combos:
        raise ValueError(f"No bound logs found for method {method} in {bounds_dir}.")

    return combos, sorted(eps_values)


def prepare_matrices(
    combos: Dict[ComboKey, Dict[float, Dict[str, float]]],
    epsilons: List[float],
    axis: str,
    depth_filter: int,
    dropout_filter: float,
    width_filter: int,
) -> Tuple[List[str], Dict[str, np.ndarray], str]:
    """Return y-axis labels, matrices, and axis label."""
    all_depths = sorted({key[0] for key in combos})
    all_dropouts = sorted({key[1] for key in combos})
    all_widths = sorted({key[2] for key in combos})

    def ensure_available(depth: int, dropout: float, width: int) -> None:
        if (depth, dropout, width) not in combos:
            raise ValueError(
                f"No entries for depth={depth}, dropout={dropout}, width={width}. "
                "Please adjust filters or rerun experiments."
            )

    if axis == "depth":
        depth_to_dropouts = {
            depth: sorted({drop for (dep, drop, _) in combos if dep == depth}) for depth in all_depths
        }
        depth_to_widths = {
            depth: sorted({width for (dep, _, width) in combos if dep == depth}) for depth in all_depths
        }
        if dropout_filter is None:
            multi = [depth for depth, drops in depth_to_dropouts.items() if len(drops) > 1]
            if multi:
                raise ValueError(
                    "Multiple dropout values per depth detected. Please set --dropout-filter."
                )
            dropout_map = {depth: drops[0] for depth, drops in depth_to_dropouts.items()}
        else:
            dropout_map = {depth: dropout_filter for depth in all_depths}
        if width_filter is None:
            multi = [depth for depth, widths in depth_to_widths.items() if len(widths) > 1]
            if multi:
                raise ValueError(
                    "Multiple hidden dimensions per depth detected. Please set --width-filter."
                )
            width_map = {depth: widths[0] for depth, widths in depth_to_widths.items()}
        else:
            width_map = {depth: width_filter for depth in all_depths}
        y_values = all_depths
        pairs = []
        for depth in y_values:
            drop = dropout_map[depth]
            width = width_map[depth]
            ensure_available(depth, drop, width)
            pairs.append((depth, drop, width))
        axis_label = "Hidden layers"
        y_tick_labels = [str(depth) for depth in y_values]
    elif axis == "dropout":
        if depth_filter is None:
            raise ValueError("Please set --depth-filter when plotting dropout axis.")
        if depth_filter not in all_depths:
            raise ValueError(f"Depth {depth_filter} not found in logs.")
        y_values = sorted({drop for (dep, drop, _) in combos if dep == depth_filter})
        if not y_values:
            raise ValueError(f"No dropout entries found for depth {depth_filter}.")
        drop_to_widths = {
            drop: sorted({width for (dep, drop2, width) in combos if dep == depth_filter and drop == drop2})
            for drop in y_values
        }
        if width_filter is None:
            multi = [drop for drop, widths in drop_to_widths.items() if len(widths) > 1]
            if multi:
                raise ValueError(
                    "Multiple hidden dimensions per dropout detected. Please set --width-filter."
                )
            width_map = {drop: widths[0] for drop, widths in drop_to_widths.items()}
        else:
            width_map = {drop: width_filter for drop in y_values}
        pairs = []
        for drop in y_values:
            width = width_map[drop]
            ensure_available(depth_filter, drop, width)
            pairs.append((depth_filter, drop, width))
        axis_label = "Dropout probability"
        y_tick_labels = [f"{drop:.2f}" for drop in y_values]
    elif axis == "width":
        if depth_filter is None:
            raise ValueError("Please set --depth-filter when plotting width axis.")
        if depth_filter not in all_depths:
            raise ValueError(f"Depth {depth_filter} not found in logs.")
        y_values = sorted({width for (dep, _, width) in combos if dep == depth_filter})
        if not y_values:
            raise ValueError(f"No width entries found for depth {depth_filter}.")
        width_to_dropouts = {
            width: sorted({drop for (dep, drop, width2) in combos if dep == depth_filter and width == width2})
            for width in y_values
        }
        if dropout_filter is None:
            multi = [width for width, drops in width_to_dropouts.items() if len(drops) > 1]
            if multi:
                raise ValueError(
                    "Multiple dropout values per width detected. Please set --dropout-filter."
                )
            dropout_map = {width: drops[0] for width, drops in width_to_dropouts.items()}
        else:
            dropout_map = {width: dropout_filter for width in y_values}
        pairs = []
        for width in y_values:
            drop = dropout_map[width]
            ensure_available(depth_filter, drop, width)
            pairs.append((depth_filter, drop, width))
        axis_label = "Hidden dimension"
        y_tick_labels = [str(width) for width in y_values]
    else:
        raise ValueError(f"Unsupported y-axis option: {axis}")

    matrices: Dict[str, np.ndarray] = {}
    for key in ["min_margin_lb", "mean_margin_lb", "certified_ratio"]:
        mat = np.full((len(y_values), len(epsilons)), np.nan)
        for i, combo in enumerate(pairs):
            entries = combos.get(combo, {})
            for j, eps in enumerate(epsilons):
                entry = entries.get(eps)
                if entry is not None:
                    mat[i, j] = entry[key]
        matrices[key] = mat

    return y_tick_labels, matrices, axis_label


def plot_heatmaps(
    y_ticks: List[str],
    epsilons: List[float],
    matrices: Dict[str, np.ndarray],
    axis_label: str,
    save_path: Path,
    show: bool,
) -> None:
    titles = {
        "min_margin_lb": "Min Margin Lower Bound",
        "mean_margin_lb": "Mean Margin Lower Bound",
        "certified_ratio": "Certified Ratio",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    x_ticks = np.arange(len(epsilons))
    y_ticks_idx = np.arange(len(y_ticks))
    for ax, key in zip(axes, ["min_margin_lb", "mean_margin_lb", "certified_ratio"]):
        data = matrices[key]
        img = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{eps:.2f}" for eps in epsilons], rotation=45)
        ax.set_yticks(y_ticks_idx)
        ax.set_yticklabels(y_ticks)
        ax.set_xlabel("epsilon (L_inf)")
        ax.set_ylabel(axis_label)
        ax.set_title(titles[key])
        cbar = fig.colorbar(img, ax=ax, shrink=0.8)
        if key == "certified_ratio":
            cbar.set_label("ratio", rotation=270, labelpad=15)
        else:
            cbar.set_label("bound value", rotation=270, labelpad=15)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved heatmaps to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot heatmaps of robustness metrics.")
    parser.add_argument(
        "--bounds-dir",
        type=str,
        default="./results/mnist_mlp_depth",
        help="Directory containing bounds_depth*.json files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="CROWN-Optimized (alpha-CROWN)",
        help="Which verification method to visualize.",
    )
    parser.add_argument(
        "--y-axis",
        choices=["depth", "dropout", "width"],
        default="depth",
        help="What to plot on the vertical axis.",
    )
    parser.add_argument(
        "--depth-filter",
        type=int,
        default=None,
        help="Depth to keep when plotting dropout/width heatmaps.",
    )
    parser.add_argument(
        "--dropout-filter",
        type=float,
        default=None,
        help="Dropout value to keep when plotting depth heatmaps.",
    )
    parser.add_argument(
        "--width-filter",
        type=int,
        default=None,
        help="Hidden dimension to keep when plotting depth/dropout heatmaps.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./results/mnist_mlp_depth/heatmaps.png",
        help="Where to save the generated heatmap figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds_dir = Path(args.bounds_dir)
    combos, epsilons = load_bounds(bounds_dir, args.method)
    y_ticks, matrices, axis_label = prepare_matrices(
        combos,
        epsilons,
        axis=args.y_axis,
        depth_filter=args.depth_filter,
        dropout_filter=args.dropout_filter,
        width_filter=args.width_filter,
    )
    plot_heatmaps(y_ticks, epsilons, matrices, axis_label, Path(args.save_path), args.show)


if __name__ == "__main__":
    main()

