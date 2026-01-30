"""Plot mobility speed sweep metrics for the MNE3SD article D analysis."""

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

ARTICLE = "article_d"
RESULTS_DIR = ROOT / "results" / "mne3sd" / ARTICLE
RESULTS_PATH = RESULTS_DIR / "mobility_speed_metrics.csv"
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
    parser.add_argument(
        "--export-eps",
        action="store_true",
        help="Export EPS files in addition to the default PNG output",
    )
    return parser.parse_args()


def resolve_export_formats(export_eps: bool) -> tuple[str, ...]:
    """Return the output formats for figure exports."""

    formats = ["png"]
    if export_eps:
        formats.append("eps")
    return tuple(formats)


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
            "jitter_s",
            "jitter_s_mean",
            "jitter_s_std",
            "energy_per_node_J",
            "energy_per_node_J_mean",
            "energy_per_node_J_std",
        ),
    )

    replicate_column = df.get("replicate")
    if replicate_column is not None:
        aggregate_mask = replicate_column.astype(str).str.lower() == "aggregate"
        aggregated = df[aggregate_mask].copy()
    else:
        aggregated = pd.DataFrame()

    if aggregated.empty:
        agg_dict: dict[str, list[str]] = {"pdr": ["mean"], "avg_delay_s": ["mean"]}
        if "jitter_s" in df.columns:
            agg_dict["jitter_s"] = ["mean", "std"]
        if "range_km" in df.columns:
            agg_dict["range_km"] = ["mean"]
        if "energy_per_node_J" in df.columns:
            agg_dict["energy_per_node_J"] = ["mean", "std"]

        grouped = df.groupby(["model", "speed_profile"], as_index=False).agg(agg_dict)
        grouped.columns = [
            column
            if isinstance(column, str)
            else "_".join(part for part in column if part)
            for column in grouped.columns
        ]

        rename_map = {"range_km_mean": "range_km"}
        aggregated = grouped.rename(columns=rename_map)

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
        aggregated["pdr_label"] = "PDR (probability)"

    aggregated["avg_delay_s_value"] = aggregated[delay_column].astype(float)

    if "jitter_s_mean" in aggregated.columns:
        aggregated["jitter_s_mean"] = aggregated["jitter_s_mean"].astype(float)
    if "jitter_s_std" in aggregated.columns:
        aggregated["jitter_s_std"] = aggregated["jitter_s_std"].astype(float)

    if "range_km" in aggregated.columns:
        aggregated["range_km"] = aggregated["range_km"].astype(float)

    if "energy_per_node_J_mean" in aggregated.columns:
        aggregated["energy_per_node_J_mean"] = aggregated["energy_per_node_J_mean"].astype(float)
    if "energy_per_node_J_std" in aggregated.columns:
        aggregated["energy_per_node_J_std"] = aggregated["energy_per_node_J_std"].astype(float)

    return aggregated


def plot_grouped_bars(
    df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_name: str,
    dpi: int,
    formats: tuple[str, ...],
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
    fig_height = 3.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pivot.plot(kind="bar", ax=ax, width=0.75)

    data = pivot.to_numpy(dtype=float)
    finite_mask = np.isfinite(data)
    max_value = data[finite_mask].max() if finite_mask.any() else 0.0

    ax.set_xlabel("Speed profile (grouped by mobility model)")
    ax.set_ylabel(ylabel)
    if ylim is None:
        if max_value > 0:
            ax.set_ylim(0, max_value * 1.1)
        else:
            ax.margins(y=0.1)
    else:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(
        title="Mobility model",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(top=0.80)
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric=output_name,
    )
    save_figure(fig, output_name, output_dir, dpi=dpi, formats=formats)
    plt.close(fig)


def plot_jitter_by_speed_profile(
    df: pd.DataFrame,
    dpi: int,
    formats: tuple[str, ...],
) -> None:
    """Plot latency jitter versus speed profile with error bars per mobility model."""

    jitter_mean_column = next(
        (
            column
            for column in ("jitter_s_mean", "jitter_s")
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )
    if jitter_mean_column is None:
        return

    jitter_std_column = next(
        (
            column
            for column in ("jitter_s_std",)
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )

    mean_pivot = df.pivot_table(
        index="speed_profile",
        columns="model_label",
        values=jitter_mean_column,
        aggfunc="mean",
    )
    if mean_pivot.empty:
        return

    mean_pivot = mean_pivot.sort_index()
    mean_pivot = mean_pivot[mean_pivot.columns.sort_values()]

    std_pivot: pd.DataFrame | None = None
    if jitter_std_column is not None:
        std_pivot = df.pivot_table(
            index="speed_profile",
            columns="model_label",
            values=jitter_std_column,
            aggfunc="mean",
        )
        std_pivot = std_pivot.reindex_like(mean_pivot)

    profiles = list(mean_pivot.index)
    x = np.arange(len(profiles), dtype=float)

    fig_width = max(4.5, 1.0 * len(profiles))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    for model_label in mean_pivot.columns:
        y = mean_pivot[model_label].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        yerr = None
        if std_pivot is not None and model_label in std_pivot.columns:
            yerr = std_pivot[model_label].to_numpy(dtype=float)
            if np.all(np.isnan(yerr)):
                yerr = None

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=model_label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_xlabel("Speed profile")
    ax.set_ylabel("Latency jitter (s)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(
        title="Mobility model",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(top=0.80)
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="latency_jitter_by_speed_profile",
    )
    save_figure(
        fig,
        "latency_jitter_by_speed_profile",
        output_dir,
        dpi=dpi,
        formats=formats,
    )
    plt.close(fig)


def _resolve_energy_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the energy mean and std columns when available."""

    energy_mean_column = next(
        (
            column
            for column in ("energy_per_node_J_mean", "energy_per_node_J")
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )
    if energy_mean_column is None:
        return None, None

    energy_std_column = next(
        (
            column
            for column in ("energy_per_node_J_std",)
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )
    return energy_mean_column, energy_std_column


def plot_energy_by_speed_profile(
    df: pd.DataFrame,
    dpi: int,
    formats: tuple[str, ...],
) -> None:
    """Plot average energy per node versus speed profile with error bars."""

    energy_mean_column, energy_std_column = _resolve_energy_columns(df)
    if energy_mean_column is None:
        return

    mean_pivot = df.pivot_table(
        index="speed_profile",
        columns="model_label",
        values=energy_mean_column,
        aggfunc="mean",
    )
    if mean_pivot.empty:
        return

    mean_pivot = mean_pivot.sort_index()
    mean_pivot = mean_pivot[mean_pivot.columns.sort_values()]

    std_pivot: pd.DataFrame | None = None
    if energy_std_column is not None:
        std_pivot = df.pivot_table(
            index="speed_profile",
            columns="model_label",
            values=energy_std_column,
            aggfunc="mean",
        )
        std_pivot = std_pivot.reindex_like(mean_pivot)

    profiles = list(mean_pivot.index)
    x = np.arange(len(profiles), dtype=float)

    fig_width = max(4.5, 1.0 * len(profiles))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    for model_label in mean_pivot.columns:
        y = mean_pivot[model_label].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        yerr = None
        if std_pivot is not None and model_label in std_pivot.columns:
            yerr = std_pivot[model_label].to_numpy(dtype=float)
            if np.all(np.isnan(yerr)):
                yerr = None

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=model_label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_xlabel("Speed profile")
    ax.set_ylabel("Average energy per node (J)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(
        title="Mobility model",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(top=0.80)
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_by_speed_profile",
    )
    save_figure(fig, "energy_by_speed_profile", output_dir, dpi=dpi, formats=formats)
    plt.close(fig)


def plot_energy_stack_by_speed_profile(
    df: pd.DataFrame,
    dpi: int,
    formats: tuple[str, ...],
) -> None:
    """Plot a stacked area chart of energy per node grouped by speed profile."""

    energy_mean_column, _ = _resolve_energy_columns(df)
    if energy_mean_column is None:
        return

    pivot = df.pivot_table(
        index="speed_profile",
        columns="model_label",
        values=energy_mean_column,
        aggfunc="mean",
    )
    if pivot.empty:
        return

    pivot = pivot.sort_index()
    pivot = pivot[pivot.columns.sort_values()]

    profiles = list(pivot.index)
    x = np.arange(len(profiles), dtype=float)

    values = np.nan_to_num(pivot.to_numpy(dtype=float), nan=0.0)

    fig_width = max(4.5, 1.0 * len(profiles))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    ax.stackplot(x, values.T, labels=pivot.columns, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_xlabel("Speed profile")
    ax.set_ylabel("Average energy per node (J)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(
        title="Mobility model",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(top=0.80)
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_stack_by_speed_profile",
    )
    save_figure(
        fig,
        "energy_stack_by_speed_profile",
        output_dir,
        dpi=dpi,
        formats=formats,
    )
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    output_name: str,
    dpi: int,
    formats: tuple[str, ...],
) -> None:
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

    plt.subplots_adjust(top=0.80)
    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric=output_name,
    )
    save_figure(fig, output_name, output_dir, dpi=dpi, formats=formats)
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

    export_formats = resolve_export_formats(args.export_eps)

    plot_grouped_bars(
        metrics,
        "pdr_percent",
        f"{pdr_label} by speed profile",
        "pdr_by_speed_profile",
        args.dpi,
        export_formats,
        pdr_format,
        ylim=pdr_ylim,
    )

    plot_grouped_bars(
        metrics,
        "avg_delay_s_value",
        "Average delay (s) by speed profile",
        "average_delay_by_speed_profile",
        args.dpi,
        export_formats,
        "{:.2f}",
    )

    plot_energy_by_speed_profile(metrics, args.dpi, export_formats)
    plot_energy_stack_by_speed_profile(metrics, args.dpi, export_formats)

    plot_jitter_by_speed_profile(metrics, args.dpi, export_formats)

    plot_heatmap(
        metrics,
        "pdr_heatmap_speed_profile_range",
        args.dpi,
        export_formats,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
