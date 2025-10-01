"""Plot mobility speed sweep metrics for the MNE3SD article B analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(ROOT))

from scripts.mne3sd.common import (
    apply_ieee_style,
    prepare_figure_directory,
    save_figure,
)

RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_b" / "mobility_speed_metrics.csv"
ARTICLE = "article_b"
SCENARIO = "mobility_speed"


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate grouped bar charts for PDR and average delay from the mobility "
            "speed sweep metrics. Optionally include a heatmap summarising PDR versus "
            "communication range when multiple ranges are present."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the mobility_speed_metrics.csv file",
    )
    parser.add_argument(
        "--style",
        help="Matplotlib style name or .mplstyle path to override the default settings",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots per inch for the exported figures (defaults to 300 dpi)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures instead of running in batch mode",
    )
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert the provided columns to numeric values in-place when present."""

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def load_metrics(path: Path) -> pd.DataFrame:
    """Return the aggregated metrics required for plotting."""

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("No rows found in the metrics CSV")

    required_columns = {"model", "speed_profile"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    _coerce_numeric(
        df,
        (
            "pdr",
            "pdr_mean",
            "avg_delay_s",
            "avg_delay_s_mean",
            "range_km",
        ),
    )

    replicate_column = df.get("replicate")
    if replicate_column is not None:
        aggregate_mask = replicate_column.astype(str).str.lower() == "aggregate"
        aggregated = df[aggregate_mask].copy()
    else:
        aggregated = pd.DataFrame()

    if aggregated.empty:
        aggregations: dict[str, str] = {"pdr": "mean", "avg_delay_s": "mean"}
        if "range_km" in df.columns:
            aggregations["range_km"] = "mean"
        grouped = df.groupby(["model", "speed_profile"], as_index=False).agg(aggregations)
        aggregated = grouped.rename(
            columns={"pdr": "pdr_mean", "avg_delay_s": "avg_delay_s_mean"}
        )

    pdr_column = next(
        (
            column
            for column in ("pdr_mean", "pdr")
            if column in aggregated.columns and aggregated[column].notna().any()
        ),
        None,
    )
    if pdr_column is None:
        raise ValueError("Unable to locate a column with PDR values")

    delay_column = next(
        (
            column
            for column in ("avg_delay_s_mean", "avg_delay_s")
            if column in aggregated.columns and aggregated[column].notna().any()
        ),
        None,
    )
    if delay_column is None:
        raise ValueError("Unable to locate a column with average delay values")

    aggregated = aggregated.copy()
    aggregated["model"] = aggregated["model"].astype(str)
    aggregated["speed_profile"] = aggregated["speed_profile"].astype(str)
    aggregated["model_label"] = aggregated["model"].str.replace("_", " ").str.title()

    pdr_values = aggregated[pdr_column].astype(float)
    if pdr_values.max() <= 1.5:
        aggregated["pdr_percent"] = pdr_values * 100.0
        aggregated["pdr_label"] = "PDR (%)"
    else:
        aggregated["pdr_percent"] = pdr_values
        aggregated["pdr_label"] = "PDR"

    aggregated["avg_delay_s_value"] = aggregated[delay_column].astype(float)

    if "range_km" in aggregated.columns:
        aggregated["range_km"] = aggregated["range_km"].astype(float)

    return aggregated


def plot_grouped_bars(
    df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_name: str,
    dpi: int,
    value_format: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot grouped bar charts where bars are separated by model."""

    pivot = df.pivot_table(
        index="speed_profile",
        columns="model_label",
        values=value_column,
        aggfunc="mean",
    )
    pivot = pivot.sort_index()
    pivot = pivot[pivot.columns.sort_values()]

    num_profiles = len(pivot.index)
    fig_width = max(4.5, 1.0 * num_profiles)
    fig_height = 3.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_xlabel("Speed profile (grouped by mobility model)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        title="Mobility model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=value_format, padding=2, fontsize=7)

    fig.subplots_adjust(right=0.8)
    fig.tight_layout()
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric=output_name,
    )
    save_figure(fig, output_name, output_dir, dpi=dpi)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, output_name: str, dpi: int) -> None:
    """Plot a heatmap of PDR versus communication range when available."""

    if "range_km" not in df.columns:
        return

    ranges = df["range_km"].dropna().unique()
    if len(ranges) <= 1:
        return

    pivot = df.pivot_table(
        index="speed_profile",
        columns="range_km",
        values="pdr_percent",
        aggfunc="mean",
    )
    pivot = pivot.sort_index()
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{col:g}" for col in pivot.columns])
    ax.set_xlabel("Communication range (km)")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_ylabel("Speed profile (rows)")

    for y, profile in enumerate(pivot.index):
        for x, rng in enumerate(pivot.columns):
            value = pivot.loc[profile, rng]
            if np.isnan(value):
                label = ""
                text_color = "white"
            elif value >= 99.95:
                label = "≈100"
                text_color = "white"
            else:
                label = f"{value:.1f}"
                text_color = "white" if value >= 50 else "black"
            ax.text(x, y, label, ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PDR (%)")

    fig.tight_layout()
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric=output_name,
    )
    save_figure(fig, output_name, output_dir, dpi=dpi)
    plt.close(fig)


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_arguments()

    apply_ieee_style()
    if args.style:
        plt.style.use(args.style)

    metrics = load_metrics(args.results)

    pdr_label = metrics["pdr_label"].iloc[0]
    pdr_ylim = (0, 105) if pdr_label.endswith("%)") else None

    pdr_format = "{:.1f}" if pdr_ylim else "{:.3f}"

    plot_grouped_bars(
        metrics,
        "pdr_percent",
        f"{pdr_label} by speed profile",
        "pdr_by_speed_profile",
        args.dpi,
        pdr_format,
        ylim=pdr_ylim,
    )

    plot_grouped_bars(
        metrics,
        "avg_delay_s_value",
        "Average delay (s) by speed profile",
        "average_delay_by_speed_profile",
        args.dpi,
        "{:.2f}",
    )

    plot_heatmap(
        metrics,
        "pdr_heatmap_speed_profile_range",
        args.dpi,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
