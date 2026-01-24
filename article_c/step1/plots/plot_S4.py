"""Trace la figure S4 (trames envoyées saturées vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    ALGO_LABELS,
    SNIR_LABELS,
    SNIR_MODES,
    apply_plot_style,
    apply_figure_layout,
    algo_label,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    load_step1_aggregated,
    metric_values,
    place_legend,
    plot_metric_by_snir,
    render_constant_metric,
    select_received_metric_key,
    resolve_percentile_keys,
    save_figure,
)

TABLE_COLUMNS = ("Algo", "SNIR", "Min", "Médiane", "Max")


def _algo_sort_key(algo: object) -> int:
    normalized = str(algo).strip().lower().replace("-", "_").replace(" ", "_")
    order = list(ALGO_LABELS.keys())
    return order.index(normalized) if normalized in order else len(order)


def _format_value(value: float) -> str:
    return f"{value:.2e}"


def _add_summary_table(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    table_rows: list[list[str]] = []
    for algo in sorted(df["algo"].dropna().unique(), key=_algo_sort_key):
        for snir_mode in SNIR_MODES:
            subset = df[(df["algo"] == algo) & (df["snir_mode"] == snir_mode)]
            if subset.empty or median_key not in subset:
                continue
            values = subset[median_key].dropna()
            if values.empty:
                continue
            table_rows.append(
                [
                    algo_label(str(algo)),
                    SNIR_LABELS[snir_mode],
                    _format_value(values.min()),
                    _format_value(values.median()),
                    _format_value(values.max()),
                ]
            )
    if not table_rows:
        return
    table = ax.table(
        cellText=table_rows,
        colLabels=TABLE_COLUMNS,
        cellLoc="center",
        loc="lower center",
        bbox=[0.0, -0.45, 1.0, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    apply_figure_layout(fig, margins={"bottom": 0.35})


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    metric_key = select_received_metric_key(rows, metric_key)
    if is_constant_metric(metric_values(rows, metric_key)):
        render_constant_metric(fig, ax)
        ax.set_title(
            "Step 1 - Sent Frames (budget saturant) vs Network size (number of nodes) "
            "(SNIR on/off)"
        )
        return fig
    plot_metric_by_snir(
        ax,
        rows,
        metric_key,
        use_algo_styles=True,
        line_width=2.4,
        marker_size=6.5,
        percentile_line_width=1.4,
    )
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Sent Frames (budget saturant, median, p10-p90)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_title(
        "Step 1 - Sent Frames (budget saturant) vs Network size (number of nodes) "
        "(SNIR on/off)"
    )
    place_legend(ax)
    _add_summary_table(fig, ax, rows, metric_key)
    return fig


def main(argv: list[str] | None = None, allow_sample: bool = True) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "sent_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S4", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
