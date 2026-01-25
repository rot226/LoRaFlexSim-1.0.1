"""Plot class load metrics for MNE3SD article A analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(ROOT))

from scripts.mne3sd.common import (
    apply_ieee_style,
    prepare_figure_directory,
    save_figure,
)
from plot_defaults import DEFAULT_FIGSIZE_SIMPLE

RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "class_load_metrics.csv"
ARTICLE = "article_a"
SCENARIO = "class_load"


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots for class load simulations, showing average energy per "
            "node and packet delivery ratio versus reporting interval."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the class_load_metrics.csv file",
    )
    parser.add_argument(
        "--style",
        help="Matplotlib style name or .mplstyle path to override the default settings",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures instead of running in batch mode",
    )
    parser.add_argument(
        "--no-energy-axis-split",
        action="store_true",
        help=(
            "Disable the automatic separation of energy curves on two axes when "
            "their magnitudes differ greatly."
        ),
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    """Read the metrics CSV, ensuring mandatory columns are present."""
    df = pd.read_csv(path)
    required = {
        "class",
        "interval_s",
        "energy_per_node_J",
        "pdr",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")
    df["interval_s"] = df["interval_s"].astype(float)
    df["energy_per_node_J"] = df["energy_per_node_J"].astype(float)
    df["pdr"] = df["pdr"].astype(float)
    return df


def plot_energy_by_interval(
    df: pd.DataFrame,
    *,
    enable_split: bool = True,
    split_threshold: float = 20.0,
) -> None:
    """Plot the average per-node energy versus interval for each class.

    When the peak energy of one class is far above the others (by default 20×), the
    plot is automatically split across two Y axes so that low-consumption classes
    remain legible while still displaying the high-consumption class.
    """

    stats = (
        df.groupby(["class", "interval_s"], as_index=False)["energy_per_node_J"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.rename(
        columns={"mean": "energy_mean", "std": "energy_std"}, inplace=True
    )
    stats["energy_std"] = stats["energy_std"].fillna(0.0)

    class_peaks = stats.groupby("class")["energy_mean"].max()
    max_peak = class_peaks.max()
    second_peak = class_peaks.sort_values(ascending=False).iloc[1] if len(class_peaks) > 1 else 0.0
    ratio = max_peak / second_peak if second_peak > 0 else float("inf") if max_peak > 0 else 1.0

    should_split = enable_split and ratio >= split_threshold and len(class_peaks) > 1

    fig, ax_low = plt.subplots(figsize=DEFAULT_FIGSIZE_SIMPLE)
    ax_high = None

    if should_split:
        cutoff = max_peak / split_threshold if split_threshold > 0 else max_peak
        low_classes = class_peaks[class_peaks <= cutoff].index.tolist()
        high_classes = class_peaks.index.difference(low_classes)
        if not low_classes:
            should_split = False
        else:
            ax_high = ax_low.twinx()
    if not should_split:
        low_classes = class_peaks.index.tolist()
        high_classes = []

    axes_for_class = {cls: ax_low for cls in low_classes}
    if ax_high is not None:
        for cls in high_classes:
            axes_for_class[cls] = ax_high

    y_limits = {ax_low: [0.0, 0.0]}
    if ax_high is not None:
        y_limits[ax_high] = [0.0, 0.0]

    for class_name, class_data in stats.groupby("class"):
        ordered = class_data.sort_values("interval_s")
        axis = axes_for_class[class_name]
        axis.errorbar(
            ordered["interval_s"],
            ordered["energy_mean"],
            yerr=ordered["energy_std"],
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

        ymin, ymax = y_limits[axis]
        ymin = min(ymin, ordered["energy_mean"].min()) if ymin else ordered["energy_mean"].min()
        ymax = max(ymax, ordered["energy_mean"].max())
        y_limits[axis] = [ymin, ymax]

    for axis, (ymin, ymax) in y_limits.items():
        if ymax == 0 and ymin == 0:
            axis.set_ylim(0, 1)
        else:
            span = ymax - ymin
            margin = 0.1 * span if span > 0 else 0.1 * ymax
            lower = 0 if ymin >= 0 else ymin - margin
            axis.set_ylim(lower, ymax + margin)

    ax_low.set_xlabel("Reporting interval (s)")
    ax_low.set_ylabel("Average energy per node (J)")
    if ax_high is not None:
        ax_high.set_ylabel("Average energy per node (J)")

    legend_handles = []
    legend_labels = []
    handles_low, labels_low = ax_low.get_legend_handles_labels()
    legend_handles.extend(handles_low)
    if should_split and ax_high is not None:
        handles_high, labels_high = ax_high.get_legend_handles_labels()
        labels_low = [f"{label} (axe gauche)" for label in labels_low]
        labels_high = [f"{label} (axe droit)" for label in labels_high]
        legend_labels.extend(labels_low)
        legend_labels.extend(labels_high)
        legend_handles.extend(handles_high)
    else:
        legend_labels.extend(labels_low)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Class",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    ax_low.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.subplots_adjust(top=0.80)

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_vs_interval",
    )
    save_figure(fig, "class_energy_vs_interval", output_dir)


def plot_pdr_by_interval(df: pd.DataFrame) -> None:
    """Plot the packet delivery ratio versus interval with error bars."""
    stats = (
        df.groupby(["class", "interval_s"], as_index=False)["pdr"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.rename(columns={"mean": "pdr_mean", "std": "pdr_std"}, inplace=True)
    stats["pdr_mean"] *= 100.0
    stats["pdr_std"] = stats["pdr_std"].fillna(0.0) * 100.0

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SIMPLE)

    for class_name, class_data in stats.groupby("class"):
        ordered = class_data.sort_values("interval_s")
        ax.errorbar(
            ordered["interval_s"],
            ordered["pdr_mean"],
            yerr=ordered["pdr_std"],
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

    ax.set_xlabel("Reporting interval (s)")
    ax.set_ylabel("PDR (%)")
    ax.set_ylim(0, 105)
    fig.legend(
        title="Class",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.subplots_adjust(top=0.80)

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="pdr_vs_interval",
    )
    save_figure(fig, "class_pdr_vs_interval", output_dir)


def main() -> None:
    args = parse_arguments()

    apply_ieee_style()
    if args.style:
        plt.style.use(args.style)

    metrics = load_metrics(args.results)

    plot_energy_by_interval(
        metrics,
        enable_split=not args.no_energy_axis_split,
    )
    plot_pdr_by_interval(metrics)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
