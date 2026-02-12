"""Trace un plot dédié throughput global (cluster all) pour ADR/MixRA-H/MixRA-Opt."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    filter_mixra_opt_fallback,
    load_step1_aggregated,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

NETWORK_SIZES = [80, 160, 320, 640, 1280]
ALGOS = ["adr", "mixra_h", "mixra_opt"]
THROUGHPUT_CANDIDATES = (
    "throughput_success_mean",
    "throughput_mean",
    "goodput_mean",
    "throughput_bps_mean",
)


def _normalize_algo(algo: object) -> str:
    return str(algo).strip().lower().replace("-", "_").replace(" ", "_")


def _select_throughput_metric(df: pd.DataFrame) -> str:
    for metric in THROUGHPUT_CANDIDATES:
        if metric in df.columns:
            series = pd.to_numeric(df[metric], errors="coerce")
            if series.notna().any():
                return metric
    raise ValueError(
        "Aucune métrique throughput trouvée. Colonnes attendues: "
        + ", ".join(THROUGHPUT_CANDIDATES)
    )


def _prepare_dataframe(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ""
    if "network_size" not in df.columns and "density" in df.columns:
        raise ValueError("Le champ network_size est requis pour ce plot dédié.")

    metric_key = _select_throughput_metric(df)
    df = df[df["network_size"].isin(NETWORK_SIZES)].copy()
    df["algo_norm"] = df["algo"].map(_normalize_algo)
    df = df[df["algo_norm"].isin(ALGOS)].copy()

    if "cluster" in df.columns and (df["cluster"] == "all").any():
        df = df[df["cluster"] == "all"].copy()

    available_snir = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    if available_snir:
        df = df[df["snir_mode"].isin(available_snir)].copy()
    else:
        df["snir_mode"] = "snir_on"

    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    grouped = (
        df.groupby(["algo_norm", "snir_mode", "network_size"], as_index=False)[metric_key]
        .mean()
        .sort_values(["snir_mode", "algo_norm", "network_size"])
    )
    return grouped, metric_key


def _metric_scale(df: pd.DataFrame, metric_key: str) -> tuple[float, str]:
    values = pd.to_numeric(df[metric_key], errors="coerce").dropna()
    if values.empty:
        return 1.0, "bit/s"
    peak = float(values.max())
    if peak >= 1000.0:
        return 1000.0, "kbit/s"
    return 1.0, "bit/s"


def _plot(df: pd.DataFrame, metric_key: str) -> plt.Figure:
    snir_modes_present = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    if not snir_modes_present:
        snir_modes_present = ["snir_on"]

    fig, axes = plt.subplots(
        1,
        len(snir_modes_present),
        figsize=resolve_ieee_figsize(len(snir_modes_present)),
        sharey=True,
    )
    if len(snir_modes_present) == 1:
        axes = [axes]

    scale, unit = _metric_scale(df, metric_key)

    for ax, snir_mode in zip(axes, snir_modes_present, strict=False):
        subset_snir = df[df["snir_mode"] == snir_mode]

        for algo in ALGOS:
            subset_algo = subset_snir[subset_snir["algo_norm"] == algo]
            points = {
                int(row.network_size): float(row[metric_key])
                for row in subset_algo.itertuples(index=False)
            }
            if not points:
                continue

            y_values = [
                points.get(size, float("nan")) / scale for size in NETWORK_SIZES
            ]
            ax.plot(
                NETWORK_SIZES,
                y_values,
                label=algo_label(algo),
                color=ALGO_COLORS.get(algo, "#4c4c4c"),
                marker=ALGO_MARKERS.get(algo, "o"),
                linestyle=SNIR_LINESTYLES.get(snir_mode, "solid"),
                linewidth=2.0,
                markersize=6,
            )

        ax.set_title(SNIR_LABELS.get(snir_mode, snir_mode))
        ax.set_xlabel("Network size (nodes)")
        ax.set_xticks(NETWORK_SIZES)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.grid(True, linestyle=":", alpha=0.35)

    axes[0].set_ylabel(f"Throughput ({unit})")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(axes) > 1:
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.78, bottom=0.2)
    else:
        axes[0].legend(loc="best", frameon=True)
        fig.subplots_adjust(bottom=0.2)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"

    rows = load_step1_aggregated(results_path, allow_sample=True)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    rows = filter_mixra_opt_fallback(rows)
    df, metric_key = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn(
            "Aucune donnée disponible pour les algos/network sizes demandés.",
            stacklevel=2,
        )
        return

    fig = _plot(df, metric_key)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S_new2_throughput_cluster_global", use_tight=False)
    assert_legend_present(fig, "plot_S_new2_throughput_cluster_global")
    plt.close(fig)


if __name__ == "__main__":
    main()
