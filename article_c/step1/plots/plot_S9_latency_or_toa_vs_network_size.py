"""Trace la figure S9 (latence/ToA vs taille du réseau, SNIR on/off)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_plot_style,
    filter_cluster,
    load_step1_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


CANDIDATE_METRICS: list[tuple[str, str]] = [
    ("toa_ms_mean", "Mean ToA (ms)"),
    ("toa_mean", "Mean ToA (ms)"),
    ("airtime_ms_mean", "Mean ToA (ms)"),
    ("airtime_mean", "Mean ToA (ms)"),
    ("latency_ms_mean", "Mean latency (ms)"),
    ("latency_mean", "Mean latency"),
    ("delay_ms_mean", "Mean latency (ms)"),
    ("delay_mean", "Mean latency"),
]


def _select_metric(rows: list[dict[str, object]]) -> tuple[str, str]:
    for key, label in CANDIDATE_METRICS:
        if any(key in row for row in rows):
            return key, label
    available_keys = {key for row in rows for key in row}
    for key in sorted(available_keys):
        if key.endswith("_mean") and ("toa" in key or "latency" in key or "delay" in key):
            label = "Mean latency"
            if "toa" in key or "airtime" in key:
                label = "Mean ToA"
            if "ms" in key:
                label = f"{label} (ms)"
            return key, label
    raise ValueError("Aucune métrique de ToA/latence trouvée dans les résultats.")


def _plot_metric(rows: list[dict[str, object]], metric_key: str, y_label: str) -> plt.Figure:
    fig, ax = plt.subplots()
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel(y_label)
    ax.set_title("Step 1 - ToA/Latency vs Network Size (SNIR on/off)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step1_aggregated(results_path), "all")

    metric_key, label = _select_metric(rows)
    fig = _plot_metric(rows, metric_key, label)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S9", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
