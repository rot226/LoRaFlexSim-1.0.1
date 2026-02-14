"""Trace throughput vs network size for clusters A/B/C and global scope."""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
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
    save_figure,
)
from article_c.step1.plots.plot_utils import load_step1_rows_with_fallback
from plot_defaults import resolve_ieee_figsize

NETWORK_SIZES = [80, 160, 320, 640, 1280]
ALGOS = ["adr", "mixra_h", "mixra_opt"]
PANEL_LABELS = ("A", "B", "C")
GLOBAL_PANEL_KEY = "__global__"
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

    if "cluster" not in df.columns:
        df["cluster"] = GLOBAL_PANEL_KEY
    df["cluster"] = df["cluster"].fillna("").astype(str)

    cluster_candidates = sorted(
        {cluster for cluster in set(df["cluster"]) if cluster and cluster != "all"}
    )
    selected_clusters = cluster_candidates[: len(PANEL_LABELS)]
    clusters_to_keep = set(selected_clusters)
    if "all" in set(df["cluster"]):
        clusters_to_keep.add("all")
    df = df[df["cluster"].isin(clusters_to_keep)].copy()

    available_snir = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    if available_snir:
        df = df[df["snir_mode"].isin(available_snir)].copy()
    else:
        df["snir_mode"] = "snir_on"

    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    grouped = (
        df.groupby(["cluster", "algo_norm", "snir_mode", "network_size"], as_index=False)[metric_key]
        .mean()
        .sort_values(["cluster", "snir_mode", "algo_norm", "network_size"])
    )

    if "all" not in set(grouped["cluster"]):
        global_df = (
            grouped.groupby(["algo_norm", "snir_mode", "network_size"], as_index=False)[
                metric_key
            ]
            .mean()
            .assign(cluster=GLOBAL_PANEL_KEY)
        )
        grouped = pd.concat([grouped, global_df], ignore_index=True)
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

    cluster_candidates = sorted(
        {cluster for cluster in set(df["cluster"]) if cluster not in {"all", GLOBAL_PANEL_KEY}}
    )
    selected_clusters = cluster_candidates[: len(PANEL_LABELS)]
    panel_order = selected_clusters + (["all"] if "all" in set(df["cluster"]) else [GLOBAL_PANEL_KEY])
    n_panels = len(panel_order)

    fig, axes = plt.subplots(1, n_panels, figsize=resolve_ieee_figsize(n_panels), sharey=True)
    if n_panels == 1:
        axes = [axes]

    scale, unit = _metric_scale(df, metric_key)

    for ax, cluster in zip(axes, panel_order, strict=False):
        subset_cluster = df[df["cluster"] == cluster]

        for algo in ALGOS:
            for snir_mode in snir_modes_present:
                subset_algo = subset_cluster[
                    (subset_cluster["algo_norm"] == algo)
                    & (subset_cluster["snir_mode"] == snir_mode)
                ]
                points = {
                    int(row.network_size): float(getattr(row, metric_key))
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
                    label=f"{algo_label(algo)} ({SNIR_LABELS.get(snir_mode, snir_mode)})",
                    color=ALGO_COLORS.get(algo, "#4c4c4c"),
                    marker=ALGO_MARKERS.get(algo, "o"),
                    linestyle=SNIR_LINESTYLES.get(snir_mode, "solid"),
                    linewidth=2.0,
                    markersize=6,
                )

        ax.set_ylabel(f"Throughput ({unit})")
        ax.set_xlabel("Network size (nodes)")
        ax.set_xticks(NETWORK_SIZES)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.grid(True, linestyle=":", alpha=0.35)


    handles: list[object] = []
    labels: list[str] = []
    for ax in axes:
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    if n_panels > 1:
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.80, bottom=0.2)
    else:
        axes[0].legend(loc="best", frameon=True)
        fig.subplots_adjust(bottom=0.2)
    return fig


def main(source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    rows = load_step1_rows_with_fallback(step_dir, allow_sample=True)
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
