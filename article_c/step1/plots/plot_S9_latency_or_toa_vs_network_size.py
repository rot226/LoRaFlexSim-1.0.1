"""Trace la figure S9 (latence/ToA vs taille du réseau, SNIR on/off)."""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    place_adaptive_legend,
    assert_legend_present,
    MetricStatus,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    metric_values,
    plot_metric_by_snir,
    render_metric_status,
    save_figure,
    warn_metric_checks_by_group,
    warn_if_insufficient_network_sizes,
)
from article_c.step1.plots.plot_utils import configure_figure, load_step1_rows_with_fallback
from plot_defaults import resolve_ieee_figsize


METRIC_KEY = "mean_toa_s"
METRIC_LABEL = "Mean ToA (s)"


def _plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    y_label: str,
    *,
    enable_suptitle: bool = False,
) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    network_sizes = sorted(df["network_size"].unique())
    warn_if_insufficient_network_sizes(network_sizes)
    warn_metric_checks_by_group(
        rows,
        metric_key,
        x_key="network_size",
        label=y_label,
        min_value=0.0,
        expected_monotonic="nondecreasing",
        group_keys=("algo", "snir_mode"),
    )
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_state, legend_handles=None)
        configure_figure(
            fig,
            ax,
            title=None,
            legend_loc="right",
            enable_suptitle=enable_suptitle,
        )
        place_adaptive_legend(fig, ax, preferred_loc="right")
        return fig
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel(y_label)
    configure_figure(
        fig,
        ax,
        title=None,
        legend_loc="right",
        enable_suptitle=enable_suptitle,
    )
    place_adaptive_legend(fig, ax, preferred_loc="right")
    return fig


def main(
    argv: list[str] | None = None,
            allow_sample: bool = True,
    enable_suptitle: bool = False, source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    parser.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Désactive le titre global (suptitle) des figures.",
    )
    args = parser.parse_args(argv)
    enable_suptitle = enable_suptitle and not args.no_suptitle
    step_dir = Path(__file__).resolve().parents[1]
    rows = load_step1_rows_with_fallback(step_dir, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    if not any(METRIC_KEY in row for row in rows):
        raise ValueError("La métrique mean_toa_s est absente des résultats.")
    fig = _plot_metric(
        rows,
        METRIC_KEY,
        METRIC_LABEL,
        enable_suptitle=enable_suptitle,
    )
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S9", use_tight=False)
    assert_legend_present(fig, "plot_S9")
    plt.close(fig)


if __name__ == "__main__":
    main()
