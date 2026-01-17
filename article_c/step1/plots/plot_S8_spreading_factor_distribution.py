"""Trace la figure S8 (distribution des SF par algorithme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_MODES,
    algo_labels,
    apply_plot_style,
    filter_cluster,
    load_step1_aggregated,
    place_legend,
    save_figure,
)


def _sf_key_candidates(sf: int) -> list[str]:
    return [
        f"sf{sf}_share_mean",
        f"sf{sf}_ratio_mean",
        f"sf{sf}_count_mean",
        f"sf{sf}_mean",
        f"sf_{sf}_share_mean",
        f"sf_{sf}_ratio_mean",
        f"sf_{sf}_count_mean",
        f"sf_{sf}_mean",
        f"sf{sf}_share",
        f"sf{sf}_ratio",
        f"sf{sf}_count",
    ]


def _extract_sf_distribution(
    row: dict[str, object],
    sf_values: list[int],
) -> dict[int, float]:
    distribution: dict[int, float] = {}
    uses_counts = False
    for sf in sf_values:
        value = 0.0
        for key in _sf_key_candidates(sf):
            if key in row:
                value = float(row.get(key, 0.0) or 0.0)
                if "count" in key:
                    uses_counts = True
                break
        distribution[sf] = value
    if not any(distribution.values()):
        return {}
    total = sum(distribution.values())
    if total > 0.0 and (uses_counts or total > 1.05):
        distribution = {sf: value / total for sf, value in distribution.items()}
    return distribution


def _aggregate_distributions(
    rows: list[dict[str, object]],
    sf_values: list[int],
) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        distribution = _extract_sf_distribution(row, sf_values)
        if not distribution:
            continue
        key = (str(row.get("algo", "")), str(row.get("snir_mode", "")))
        if key not in grouped:
            grouped[key] = {
                "count": 0,
                "values": {sf: 0.0 for sf in sf_values},
            }
        grouped[key]["count"] = int(grouped[key]["count"]) + 1
        values: dict[int, float] = grouped[key]["values"]
        for sf, share in distribution.items():
            values[sf] += share

    aggregated: dict[tuple[str, str], dict[int, float]] = {}
    for key, payload in grouped.items():
        count = int(payload["count"])
        values: dict[int, float] = payload["values"]
        if count <= 0:
            continue
        aggregated[key] = {sf: value / count for sf, value in values.items()}
    return aggregated


def _plot_distribution(rows: list[dict[str, object]]) -> plt.Figure:
    sf_values = list(DEFAULT_CONFIG.radio.spreading_factors)
    snir_modes = [mode for mode in SNIR_MODES if any(row.get("snir_mode") == mode for row in rows)]
    if not snir_modes:
        snir_modes = sorted({row.get("snir_mode", "") for row in rows})
    algorithms = sorted({row.get("algo", "") for row in rows})
    distribution_by_group = _aggregate_distributions(rows, sf_values)

    fig, axes = plt.subplots(1, len(snir_modes), figsize=(6 * len(snir_modes), 4), sharey=True)
    if len(snir_modes) == 1:
        axes = [axes]

    colors = [plt.get_cmap("viridis")(idx / max(1, len(sf_values) - 1)) for idx in range(len(sf_values))]
    x_positions = list(range(len(algorithms)))

    for ax, snir_mode in zip(axes, snir_modes, strict=False):
        bottoms = [0.0 for _ in algorithms]
        for sf_idx, sf in enumerate(sf_values):
            heights = [
                distribution_by_group.get((algo, snir_mode), {}).get(sf, 0.0)
                for algo in algorithms
            ]
            ax.bar(
                x_positions,
                heights,
                bottom=bottoms,
                color=colors[sf_idx],
                label=f"SF{sf}",
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights, strict=False)]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(algo_labels(algorithms))
        ax.set_xlabel("Algorithm")
        ax.set_title(SNIR_LABELS.get(snir_mode, snir_mode))
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Share of nodes")
    place_legend(axes[-1])
    fig.suptitle("Step 1 - Spreading Factor Distribution (SNIR on/off)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step1_aggregated(results_path), "all")

    fig = _plot_distribution(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S8", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
