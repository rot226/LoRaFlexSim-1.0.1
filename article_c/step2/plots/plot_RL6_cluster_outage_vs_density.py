"""Trace la figure RL6 (outage par cluster vs densité)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    create_right_legend_layout,
    algo_label,
    apply_plot_style,
    apply_figure_layout,
    assert_legend_present,
    clear_axis_legends,
    collect_legend_entries,
    MetricStatus,
    deduplicate_legend_entries,
    ensure_network_size,
    filter_rows_by_network_sizes,
    fallback_legend_handles,
    is_constant_metric,
    legend_handles_for_algos_snir,
    load_step2_aggregated,
    metric_values,
    normalize_network_size_rows,
    render_metric_status,
    save_figure,
    suptitle_y_from_top,
)
from plot_defaults import resolve_ieee_figsize

ALGO_ALIASES = {
    "adr": "adr",
    "loba": "loba",
    "lo_ba": "loba",
    "lora_baseline": "loba",
    "lorawan_baseline": "loba",
    "mixra_h": "mixra_h",
    "mixra_opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
}
TARGET_ALGOS = {"adr", "loba", "mixra_h", "mixra_opt", "ucb1_sf"}
RIGHT_LEGEND_WIDTH_FACTOR = 1.3
RIGHT_LEGEND_RATIO_FACTOR = 1.15


def _right_legend_figsize(cluster_count: int) -> tuple[float, float]:
    width, height = resolve_ieee_figsize(cluster_count)
    return (width * RIGHT_LEGEND_WIDTH_FACTOR, height)


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes:
        return None
    return network_sizes


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        print(
            "ERREUR: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée."
        )
        return True
    return False


def _title_suffix(network_sizes: list[int]) -> str:
    if len(network_sizes) == 1:
        return " (taille unique)"
    return ""


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    return {cluster: f"C{idx + 1}" for idx, cluster in enumerate(clusters)}


def _normalize_algo_label(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_algo(algo: str) -> str | None:
    normalized = _normalize_algo_label(algo)
    if not normalized:
        return None
    return ALGO_ALIASES.get(normalized, normalized)


def _label_for_algo(algo: str) -> str:
    canonical = _canonical_algo(algo)
    if canonical is None:
        return algo
    return algo_label(canonical)


def _extract_success_rate(row: dict[str, object]) -> float | None:
    for key in ("success_rate_mean", "success_rate", "success_mean"):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        return float(value)
    return None


def _outage_probability(row: dict[str, object]) -> float:
    success_rate = _extract_success_rate(row)
    if success_rate is None:
        success_rate = 0.0
    outage = 1.0 - success_rate
    return max(0.0, min(1.0, outage))


def _with_outage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["outage_prob"] = _outage_probability(row)
        enriched.append(enriched_row)
    return enriched


def _filter_algorithms(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized_labels = {
        _canonical_algo(str(row.get("algo", "")))
        for row in rows
        if row.get("algo") is not None
    }
    normalized_labels.discard(None)
    allowed = TARGET_ALGOS | normalized_labels
    filtered = [
        row for row in rows if _canonical_algo(str(row.get("algo", ""))) in allowed
    ]
    return filtered or rows


def _plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int] | None,
) -> plt.Figure | None:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    if network_sizes is None:
        network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    single_size = len(network_sizes) == 1
    only_size = network_sizes[0] if single_size else None
    available_clusters = {
        row["cluster"] for row in rows if row.get("cluster") not in (None, "all")
    }
    clusters = [
        cluster
        for cluster in DEFAULT_CONFIG.qos.clusters
        if cluster in available_clusters
    ]
    if not clusters:
        clusters = sorted(available_clusters)
    if not clusters:
        warnings.warn(
            "Aucun cluster disponible après filtrage; aucune figure ne sera tracée.",
            stacklevel=2,
        )
        return None
    cluster_labels = _cluster_labels(clusters)

    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    apply_figure_layout(fig, figsize=_right_legend_figsize(len(clusters)))
    if len(clusters) == 1:
        axes = [axes]

    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            axes,
            metric_state,
            show_fallback_legend=True,
            legend_handles=legend_handles_for_algos_snir(["snir_on"]),
        )
        return fig

    algorithms = sorted({row["algo"] for row in rows})
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        for algo in algorithms:
            points = {
                int(row["network_size"]): row[metric_key]
                for row in cluster_rows
                if row.get("algo") == algo
            }
            if not points:
                continue
            if single_size:
                value = points.get(only_size)
                if value is None or pd.isna(value):
                    continue
                ax.scatter(
                    [only_size],
                    [value],
                    label=_label_for_algo(str(algo)),
                )
                continue
            values = [points.get(size, float("nan")) for size in network_sizes]
            ax.plot(network_sizes, values, marker="o", label=_label_for_algo(str(algo)))
        ax.set_xlabel("Network size (nodes)")
        ax.set_ylabel("Outage (prob.)")
        ax.set_xticks(network_sizes)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    clear_axis_legends(axes)
    handles, labels = collect_legend_entries(axes)
    handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        handles, labels = legend_handles_for_algos_snir(["snir_on"])
    if not handles:
        handles, labels = fallback_legend_handles()
    create_right_legend_layout(
        fig,
        axes,
        handles=handles if handles else None,
        labels=labels if handles else None,
        extra_width_factor=RIGHT_LEGEND_RATIO_FACTOR,
    )
    return fig


def _resolve_intermediate_path(results_path: Path) -> Path | None:
    by_replication = results_path.with_name("aggregated_results_by_replication.csv")
    if by_replication.exists():
        return by_replication
    by_round = results_path.with_name("aggregated_results_by_round.csv")
    if by_round.exists():
        return by_round
    return None


def _load_step2_intermediate(results_path: Path) -> list[dict[str, object]]:
    intermediate_path = _resolve_intermediate_path(results_path)
    if intermediate_path is None:
        return []
    df = pd.read_csv(intermediate_path)
    rows = df.to_dict("records")
    numeric_columns = {
        "network_size",
        "density",
        "success_rate",
        "success_rate_mean",
        "success_mean",
        "collision_norm",
        "round",
        "replication",
    }
    for row in rows:
        for key in numeric_columns:
            if key in row and row[key] not in (None, ""):
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = None
    return rows


def _replication_key(rows: list[dict[str, object]]) -> str | None:
    if any(row.get("replication") not in (None, "") for row in rows):
        return "replication"
    if any(row.get("round") not in (None, "") for row in rows):
        return "round"
    return None


def _log_outage_diagnostics(rows: list[dict[str, object]]) -> None:
    success_rates = [
        rate for row in rows if (rate := _extract_success_rate(row)) is not None
    ]
    collision_norms = [
        float(row["collision_norm"])
        for row in rows
        if row.get("collision_norm") not in (None, "")
    ]
    if success_rates:
        print(
            "Diagnostic outage (success_rate) - "
            f"min={min(success_rates):.4f}, max={max(success_rates):.4f}"
        )
    if collision_norms:
        print(
            "Diagnostic outage (collision_norm) - "
            f"min={min(collision_norms):.4f}, max={max(collision_norms):.4f}"
        )


def _plot_raw_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int] | None,
) -> plt.Figure | None:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    if network_sizes is None:
        network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    replication_key = _replication_key(rows)
    if replication_key is None:
        warnings.warn(
            "Aucune colonne replication/round disponible pour la métrique brute.",
            stacklevel=2,
        )
        return None
    available_clusters = {
        row["cluster"] for row in rows if row.get("cluster") not in (None, "all")
    }
    clusters = [
        cluster
        for cluster in DEFAULT_CONFIG.qos.clusters
        if cluster in available_clusters
    ]
    if not clusters:
        clusters = sorted(available_clusters)
    if not clusters:
        warnings.warn(
            "Aucun cluster disponible pour la métrique brute.",
            stacklevel=2,
        )
        return None
    cluster_labels = _cluster_labels(clusters)
    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    apply_figure_layout(fig, figsize=_right_legend_figsize(len(clusters)))
    if len(clusters) == 1:
        axes = [axes]
    algorithms = sorted({row["algo"] for row in rows})
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        for algo in algorithms:
            algo_rows = [row for row in cluster_rows if row.get("algo") == algo]
            replications = sorted(
                {
                    row.get(replication_key)
                    for row in algo_rows
                    if row.get(replication_key) not in (None, "")
                }
            )
            for rep_idx, replication in enumerate(replications):
                points = {
                    int(row["network_size"]): row[metric_key]
                    for row in algo_rows
                    if row.get(replication_key) == replication
                }
                if not points:
                    continue
                values = [points.get(size, float("nan")) for size in network_sizes]
                label = _label_for_algo(str(algo)) if rep_idx == 0 else None
                ax.plot(
                    network_sizes,
                    values,
                    marker="o",
                    alpha=0.35,
                    label=label,
                )
        ax.set_xlabel("Network size (nodes)")
        ax.set_ylabel("Outage (raw, prob.)")
        ax.set_xticks(network_sizes)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    clear_axis_legends(axes)
    handles, labels = collect_legend_entries(axes)
    handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        handles, labels = legend_handles_for_algos_snir(["snir_on"])
    if not handles:
        handles, labels = fallback_legend_handles()
    create_right_legend_layout(
        fig,
        axes,
        handles=handles if handles else None,
        labels=labels if handles else None,
        extra_width_factor=RIGHT_LEGEND_RATIO_FACTOR,
    )
    return fig


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Trace en plus la métrique brute par cluster et réplication.",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)
    rows = _filter_algorithms(rows)
    rows = _with_outage(rows)

    fig = _plot_metric(rows, "outage_prob", network_sizes_filter)
    if fig is None:
        return
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL6_cluster_outage_vs_density", use_tight=False)
    assert_legend_present(fig, "plot_RL6_cluster_outage_vs_density")
    plt.close(fig)

    if args.debug_raw:
        raw_rows = _load_step2_intermediate(results_path)
        if not raw_rows:
            warnings.warn(
                "Fichier intermédiaire manquant pour la métrique brute.",
                stacklevel=2,
            )
            return
        raw_rows = [row for row in raw_rows if row.get("cluster") != "all"]
        raw_rows = [row for row in raw_rows if row.get("snir_mode") == "snir_on"]
        normalize_network_size_rows(raw_rows)
        raw_rows, _ = filter_rows_by_network_sizes(raw_rows, network_sizes_filter)
        raw_rows = _filter_algorithms(raw_rows)
        raw_rows = _with_outage(raw_rows)
        _log_outage_diagnostics(raw_rows)
        raw_fig = _plot_raw_metric(raw_rows, "outage_prob", network_sizes_filter)
        if raw_fig is None:
            return
        save_figure(
            raw_fig,
            output_dir,
            "plot_RL6_cluster_outage_raw_by_replication",
            use_tight=False,
        )
        assert_legend_present(raw_fig, "plot_RL6_cluster_outage_raw_by_replication")
        plt.close(raw_fig)


if __name__ == "__main__":
    main()
