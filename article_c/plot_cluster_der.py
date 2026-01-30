"""Trace le DER par cluster à partir des CSV agrégés."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_LABELS,
    ALGO_MARKERS,
    ALGO_ALIASES,
    add_global_legend,
    apply_figure_layout,
    apply_plot_style,
    assert_legend_present,
    filter_rows_by_network_sizes,
    legend_margins,
    load_step1_aggregated,
    load_step2_aggregated,
    parse_export_formats,
    save_figure,
    set_default_export_formats,
    set_network_size_ticks,
)
from plot_defaults import resolve_ieee_figsize

PREFERRED_ALGOS = (
    "apra",
    "aimi",
    "mixra_h",
    "mixra_opt",
    "adr",
    "loba",
    "ucb1_sf",
)


def _normalize_algo(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALGO_ALIASES.get(normalized, normalized)


def _load_aggregated_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = (
        base_dir / "step1" / "results" / "aggregated_results.csv",
        base_dir / "step2" / "results" / "aggregated_results.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.parts[-3] == "step2":
                rows.extend(load_step2_aggregated(path))
            else:
                rows.extend(load_step1_aggregated(path))
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
    return rows


def _resolve_der_source(rows: list[dict[str, object]]) -> tuple[str, str]:
    for key in ("der_mean", "der", "der_p50", "der_p90", "der_p10"):
        if any(key in row for row in rows):
            return "direct", key
    for key in ("pdr_mean", "pdr", "pdr_p50", "pdr_p90", "pdr_p10"):
        if any(key in row for row in rows):
            return "pdr", key
    raise ValueError("Impossible de trouver une colonne DER/PDR dans les CSV.")


def _prepare_dataframe(
    rows: list[dict[str, object]],
    der_mode: str,
    metric_key: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        cluster = str(row.get("cluster", "")).strip().lower()
        if not cluster:
            continue
        algo = _normalize_algo(row.get("algo"))
        network_size = row.get("network_size")
        if network_size in (None, ""):
            continue
        try:
            network_size_value = int(round(float(network_size)))
        except (TypeError, ValueError):
            continue
        metric_value = row.get(metric_key)
        if metric_value in (None, ""):
            continue
        try:
            metric_value_float = float(metric_value)
        except (TypeError, ValueError):
            continue
        der_value = metric_value_float
        if der_mode == "pdr":
            der_value = 1.0 - metric_value_float
        if math.isnan(der_value):
            continue
        records.append(
            {
                "cluster": cluster,
                "algo": algo,
                "network_size": network_size_value,
                "der": der_value,
            }
        )
    if not records:
        return pd.DataFrame(columns=["cluster", "algo", "network_size", "der"])
    df = pd.DataFrame.from_records(records)
    return (
        df.groupby(["cluster", "algo", "network_size"], as_index=False)["der"]
        .mean()
        .sort_values(["cluster", "algo", "network_size"])
    )


def _select_clusters(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    available = sorted(set(df["cluster"]))
    if requested:
        normalized = [cluster.strip().lower() for cluster in requested if cluster]
        return [cluster for cluster in normalized if cluster in available]
    ordered = [
        cluster
        for cluster in DEFAULT_CONFIG.qos.clusters
        if cluster in available
    ]
    return ordered or available


def _select_algorithms(df: pd.DataFrame) -> list[str]:
    available = sorted(set(df["algo"]))
    ordered = [algo for algo in PREFERRED_ALGOS if algo in available]
    remaining = [algo for algo in available if algo not in ordered]
    return ordered + remaining


def _plot_der_by_cluster(df: pd.DataFrame, clusters: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    apply_figure_layout(fig, figsize=resolve_ieee_figsize(len(clusters)))
    if len(clusters) == 1:
        axes = [axes]
    algo_order = _select_algorithms(df)

    legend_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []
    seen_algos: set[str] = set()

    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_df = df[df["cluster"] == cluster]
        network_sizes = sorted(set(cluster_df["network_size"]))
        if not network_sizes:
            continue
        for algo in algo_order:
            algo_df = cluster_df[cluster_df["algo"] == algo]
            if algo_df.empty:
                continue
            values = [
                algo_df.loc[algo_df["network_size"] == size, "der"].mean()
                if size in set(algo_df["network_size"])
                else float("nan")
                for size in network_sizes
            ]
            label = ALGO_LABELS.get(algo, algo)
            (line,) = ax.plot(
                network_sizes,
                values,
                marker=ALGO_MARKERS.get(algo, "o"),
                color=ALGO_COLORS.get(algo),
                label=label,
            )
            if algo not in seen_algos:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_algos.add(algo)
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel("DER")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        set_network_size_ticks(ax, network_sizes)

    add_global_legend(
        fig,
        axes[0],
        legend_loc="above",
        handles=legend_handles,
        labels=legend_labels,
        use_fallback=False,
    )
    apply_figure_layout(fig, margins=legend_margins("above"))
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        help="Formats d'export (ex: png,pdf,eps).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Dossier de sortie pour les figures.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        help="Filtrer les clusters (ex: --clusters gold silver).",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)

    apply_plot_style()
    export_formats = parse_export_formats(args.formats)
    set_default_export_formats(export_formats)

    base_dir = Path(__file__).resolve().parent
    rows = _load_aggregated_rows(base_dir)
    if not rows:
        warnings.warn("Aucun CSV agrégé trouvé.", stacklevel=2)
        return

    rows = [row for row in rows if str(row.get("cluster", "")).lower() != "all"]
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    if not rows:
        warnings.warn("Aucune donnée après filtrage.", stacklevel=2)
        return

    der_mode, metric_key = _resolve_der_source(rows)
    df = _prepare_dataframe(rows, der_mode, metric_key)
    if df.empty:
        warnings.warn("Aucune donnée DER exploitable.", stacklevel=2)
        return

    clusters = _select_clusters(df, args.clusters)
    if not clusters:
        warnings.warn("Aucun cluster correspondant.", stacklevel=2)
        return

    fig = _plot_der_by_cluster(df, clusters)
    output_dir = args.output_dir or (base_dir / "plots" / "output")
    save_figure(fig, output_dir, "plot_cluster_der", use_tight=False)
    assert_legend_present(fig, "plot_cluster_der")
    plt.close(fig)


if __name__ == "__main__":
    main()
