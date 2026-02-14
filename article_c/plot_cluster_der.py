"""Trace le DER par cluster à partir des CSV agrégés."""

from __future__ import annotations

import argparse
import math
import logging
from pathlib import Path
from collections.abc import Callable
import sys
import warnings
from importlib.util import find_spec

import matplotlib.pyplot as plt
import pandas as pd


if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

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
from article_c.common.plotting_style import label_for

LOGGER = logging.getLogger(__name__)
LAST_EFFECTIVE_SOURCE = "aggregates"

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


def _load_rows_from_paths(
    paths: list[Path],
    *,
    loader: Callable[[Path], list[dict[str, object]]],
    step_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            rows.extend(loader(path))
        except (OSError, ValueError) as exc:
            warnings.warn(f"CSV {step_label} ignoré ({path}): {exc}", stacklevel=2)
    return rows


def _load_aggregated_rows(base_dir: Path, *, source: str) -> tuple[list[dict[str, object]], str]:
    normalized_source = str(source).strip().lower()
    if normalized_source not in {"aggregates", "by_size"}:
        raise ValueError("Source CSV non supportée. Utilisez aggregates ou by_size.")

    rows: list[dict[str, object]] = []
    used_sources: set[str] = set()
    candidates = (
        (1, load_step1_aggregated),
        (2, load_step2_aggregated),
    )

    for step, loader in candidates:
        results_dir = base_dir / f"step{step}" / "results"
        aggregate_path = results_dir / "aggregates" / "aggregated_results.csv"
        step_label = f"Step{step}"
        current: list[dict[str, object]] = []

        if normalized_source == "aggregates" and aggregate_path.exists():
            current = _load_rows_from_paths([aggregate_path], loader=loader, step_label=step_label)
            if current:
                used_sources.add("aggregates")

        if not current:
            rep_paths = sorted(results_dir.glob("by_size/size_*/rep_*/aggregated_results.csv"))
            if normalized_source == "by_size" and not rep_paths:
                rep_paths = sorted(results_dir.glob("by_size/size_*/aggregated_results.csv"))
            current = _load_rows_from_paths(rep_paths, loader=loader, step_label=step_label)
            if current:
                used_sources.add("by_size")

        if current:
            rows.extend(current)

    effective_source = normalized_source if normalized_source in used_sources else "mixed"
    if "by_size" in used_sources and "aggregates" in used_sources:
        effective_source = "mixed"
    elif len(used_sources) == 1:
        effective_source = next(iter(used_sources))
    LOGGER.info("source utilisée: %s", effective_source)
    return rows, effective_source


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
        ax.set_xlabel(label_for("x.network_size"))
        ax.set_ylabel(label_for("y.der"))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        set_network_size_ticks(ax, network_sizes)

    add_global_legend(
        fig,
        axes[0],
        legend_loc="right",
        handles=legend_handles,
        labels=legend_labels,
        use_fallback=False,
    )
    apply_figure_layout(fig, margins=legend_margins("right"), legend_loc="right")
    return fig


def main(
    argv: list[str] | None = None,
    *,
    close_figures: bool = True,
    source: str | None = None,
) -> None:
    global LAST_EFFECTIVE_SOURCE
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        help="Formats d'export (ex: png,eps).",
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
    parser.add_argument(
        "--source",
        choices=("aggregates", "by_size"),
        default="aggregates",
        help="Source CSV à utiliser: agrégat global ou lecture by_size.",
    )
    args = parser.parse_args(argv)

    if source is not None:
        args.source = str(source).strip().lower()

    apply_plot_style()
    export_formats = parse_export_formats(args.formats)
    set_default_export_formats(export_formats)

    base_dir = Path(__file__).resolve().parent
    rows, effective_source = _load_aggregated_rows(base_dir, source=args.source)
    LAST_EFFECTIVE_SOURCE = effective_source
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
    if close_figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
