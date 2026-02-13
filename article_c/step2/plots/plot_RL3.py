"""Trace la figure RL3 (débit réussi médian vs densité)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    assert_legend_present,
    MetricStatus,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    is_constant_metric,
    load_step2_aggregated,
    metric_values,
    normalize_network_size_rows,
    place_adaptive_legend,
    legend_handles_for_algos_snir,
    plot_metric_by_algo,
    render_metric_status,
    save_figure,
    warn_metric_checks_by_group,
)
from article_c.common.plotting_style import label_for
from plot_defaults import RL_FIGURE_SCALE, resolve_ieee_figsize

ALGO_ALIASES = {
    "adr": "adr",
    "mixra_h": "mixra_h",
    "mixra_opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "ucb1-sf": "ucb1_sf",
}
COMMON_CURVE_LABELS = {
    "adr": "ADR",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
    "ucb1_sf": "UCB1-SF",
}
SNIR_MODE_LABELS = {"snir_on": "SNIR on", "snir_off": "SNIR off"}


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


def _canonical_algo(algo: object) -> str:
    normalized = str(algo or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALGO_ALIASES.get(normalized, normalized)


def _label_for_algo(algo: object) -> str:
    canonical = _canonical_algo(algo)
    if canonical in COMMON_CURVE_LABELS:
        return COMMON_CURVE_LABELS[canonical]
    return algo_label(canonical) if canonical else str(algo)


def _plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int] | None,
) -> plt.Figure | None:
    df = pd.DataFrame(rows)
    if "algo" in df.columns:
        algo_col = "algo"
    elif "algorithm" in df.columns:
        algo_col = "algorithm"
    else:
        algo_col = None
    series_count = len(df[algo_col].dropna().unique()) if algo_col else None
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count, scale=RL_FIGURE_SCALE))
    ensure_network_size(rows)
    if network_sizes is None:
        network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    warn_metric_checks_by_group(
        rows,
        metric_key,
        x_key="network_size",
        label="Successful throughput",
        min_value=0.0,
        expected_monotonic="nonincreasing",
        group_keys=("cluster", "algo", "snir_mode"),
    )
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            ax,
            metric_state,
            show_fallback_legend=True,
            legend_handles=legend_handles_for_algos_snir(["snir_on"]),
        )
        return fig
    plot_metric_by_algo(
        ax,
        rows,
        metric_key,
        network_sizes,
        label_fn=lambda algo: _label_for_algo(algo),
        snir_label_fn=lambda mode: SNIR_MODE_LABELS.get(str(mode), str(mode)),
    )
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel(label_for("y.successful_throughput"))
    place_adaptive_legend(fig, ax, preferred_loc="right")
    return fig


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregates" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)

    fig = _plot_metric(rows, "throughput_success_mean", network_sizes_filter)
    if fig is None:
        return
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL3", use_tight=False)
    assert_legend_present(fig, "plot_RL3")
    plt.close(fig)


if __name__ == "__main__":
    main()
