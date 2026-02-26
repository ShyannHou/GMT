
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def parse_results(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    metrics: Dict[str, Dict[str, list]] = {}
    current = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        token = line.replace(",", "").strip()
        if token in ("acc", "bac", "f1"):
            current = token
            metrics[current] = {"mean": [], "std": []}
            continue
        if current is None:
            continue

        means = []
        stds = []
        for cell in line.split(","):
            cell = cell.strip()
            if not cell:
                continue
            cell = cell.replace("%", "")
            if "\u00b1" in cell:
                mean_s, std_s = cell.split("\u00b1", 1)
            else:
                mean_s, std_s = cell, "nan"
            means.append(float(mean_s))
            stds.append(float(std_s))

        metrics[current]["mean"].append(means)
        metrics[current]["std"].append(stds)

    parsed: Dict[str, Dict[str, np.ndarray]] = {}
    for name, data in metrics.items():
        mean = np.array(data["mean"], dtype=float)
        std = np.array(data["std"], dtype=float)
        if mean.ndim != 2 or mean.shape != std.shape:
            raise ValueError(f"Malformed data for {name}: mean {mean.shape}, std {std.shape}")
        if mean.shape[0] != mean.shape[1]:
            raise ValueError(f"Expected square matrix for {name}, got {mean.shape}")
        parsed[name] = {"mean": mean, "std": std}
    return parsed

def plot_metric(
    mean: np.ndarray,
    std: np.ndarray,
    metric_label: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    num_total = mean.shape[0]
    if num_total < 2:
        raise ValueError("Need at least 2x2 matrix to plot.")
    num_task = num_total - 1

    perf = mean[:num_task, :num_task]
    perf_std = std[:num_task, :num_task]
    avg_perf = mean[:num_task, -1]
    avg_perf_std = std[:num_task, -1]
    forget = mean[-1, :num_task]
    forget_std = std[-1, :num_task]
    avg_forget = mean[-1, -1]
    final_avg = mean[num_task - 1, -1]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig_w = 4.8 + 0.25 * num_task
    fig_h = 4.4 + 0.22 * num_task
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2,
        3,
        width_ratios=[1.0, 0.06, 0.35],
        height_ratios=[1.0, 0.35],
        wspace=0.25,
        hspace=0.25,
        figure=fig,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2], sharey=ax)
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_corner = fig.add_subplot(gs[1, 2])
    ax_corner.axis("off")
    ax_unused = fig.add_subplot(gs[1, 1])
    ax_unused.axis("off")

    vmin, vmax = perf.min(), perf.max()
    im = ax.imshow(perf, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{metric_label} (%)")

    ax.set_xticks(np.arange(num_task))
    ax.set_yticks(np.arange(num_task))
    ax.set_xticklabels([str(i + 1) for i in range(num_task)])
    ax.set_yticklabels([str(i + 1) for i in range(num_task)])
    ax.set_xlabel("")
    ax.set_ylabel("Training task")
    ax.tick_params(length=0, labelbottom=False)
    ax.set_xticks(np.arange(-0.5, num_task, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_task, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.7)

    if title:
        ax.set_title(title)

    y = np.arange(num_task)
    ax_right.errorbar(
        avg_perf,
        y,
        xerr=avg_perf_std,
        fmt="o",
        color="#3b4a6b",
        ecolor="#3b4a6b",
        elinewidth=0.8,
        capsize=2,
    )
    ax_right.set_xlabel("Avg (%)")
    ax_right.grid(axis="x", color="0.9", linewidth=0.7)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    pad = max(1.0, 0.03 * (vmax - vmin))
    ax_right.set_xlim(vmin - pad, vmax + pad)

    x = np.arange(num_task)
    ax_bottom.bar(
        x,
        forget,
        yerr=forget_std,
        color="#7a869a",
        edgecolor="none",
        capsize=2,
    )
    ax_bottom.axhline(avg_forget, color="0.3", linewidth=1, linestyle="--")
    ax_bottom.set_ylabel("Forgetting (%)")
    ax_bottom.set_xlabel("Test task")
    ax_bottom.grid(axis="y", color="0.9", linewidth=0.7)
    ax_bottom.set_xticks(np.arange(num_task))
    ax_bottom.set_xticklabels([str(i + 1) for i in range(num_task)])

    ax_corner.text(
        0.0,
        0.8,
        f"Final avg: {final_avg:.2f}%",
        fontsize=9,
        ha="left",
        va="center",
    )
    ax_corner.text(
        0.0,
        0.45,
        f"Avg forgetting: {avg_forget:.2f}%",
        fontsize=9,
        ha="left",
        va="center",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot continual learning results as a paper-style figure.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/DBLP/DYGRA_GCN_DBLP_reduction_0.5.csv"),
        help="Path to the results CSV.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc", "bac", "f1"],
        help="Which metric to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/DBLP/figures/DYGRA_GCN_DBLP_reduction_0.5_acc.pdf"),
        help="Output figure path (pdf/png/svg).",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = parser.parse_args()

    data = parse_results(args.input)
    if args.metric not in data:
        raise ValueError(f"Metric {args.metric} not found in {args.input}")
    metric_label = {"acc": "Accuracy", "bac": "Balanced accuracy", "f1": "Macro-F1"}[args.metric]
    plot_metric(data[args.metric]["mean"], data[args.metric]["std"], metric_label, args.output, args.title)

if __name__ == "__main__":
    main()
