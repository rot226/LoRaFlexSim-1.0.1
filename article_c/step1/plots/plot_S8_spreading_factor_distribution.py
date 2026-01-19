"""Trace la figure S8 (distribution des SF par algorithme)."""

from __future__ import annotations

import csv
import logging
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


def _read_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_sf_selected(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed


def _normalize_algo(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized


def _normalize_snir_mode(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized in {"on", "snir_on"}:
        return "snir_on"
    if normalized in {"off", "snir_off"}:
        return "snir_off"
    return normalized


def _aggregate_sf_selected(
    rows: list[dict[str, str]],
    sf_values: list[int],
) -> dict[tuple[str, str], dict[int, float]]:
    counts: dict[tuple[str, str], dict[int, int]] = {}
    for row in rows:
        sf_value = _parse_sf_selected(row.get("sf_selected"))
        if sf_value is None or sf_value not in sf_values:
            continue
        algo_raw = (row.get("algo") or "").strip()
        snir_raw = (row.get("snir_mode") or "").strip()
        algo = _normalize_algo(algo_raw) or algo_raw
        snir_mode = _normalize_snir_mode(snir_raw) or snir_raw
        if not algo or not snir_mode:
            continue
        key = (algo, snir_mode)
        if key not in counts:
            counts[key] = {sf: 0 for sf in sf_values}
        counts[key][sf_value] += 1

    aggregated: dict[tuple[str, str], dict[int, float]] = {}
    for key, sf_counts in counts.items():
        total = sum(sf_counts.values())
        if total <= 0:
            continue
        aggregated[key] = {sf: count / total for sf, count in sf_counts.items()}
    return aggregated


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
    extra_snir_modes = [
        mode
        for mode in sorted({row.get("snir_mode", "") for row in rows})
        if mode and mode not in snir_modes
    ]
    snir_modes = snir_modes + extra_snir_modes
    if not snir_modes:
        snir_modes = sorted({row.get("snir_mode", "") for row in rows if row.get("snir_mode")})
    algorithms = sorted({row.get("algo", "") for row in rows if row.get("algo")})
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
    logger = logging.getLogger(__name__)
    step_dir = Path(__file__).resolve().parents[1]
    raw_results_path = step_dir / "results" / "raw_results.csv"
    aggregated_results_path = step_dir / "results" / "aggregated_results.csv"
    sf_values = list(DEFAULT_CONFIG.radio.spreading_factors)
    raw_rows = _read_raw_rows(raw_results_path)
    sf_rows = [row for row in raw_rows if _parse_sf_selected(row.get("sf_selected")) is not None]
    distribution_by_group: dict[tuple[str, str], dict[int, float]] = {}
    if sf_rows:
        distribution_by_group = _aggregate_sf_selected(sf_rows, sf_values)
        if not distribution_by_group:
            logger.warning(
                "Les filtres S8 ont supprimé toutes les lignes: utilisation des résultats agrégés."
            )
    elif raw_rows:
        logger.warning(
            "Aucune ligne sf_selected détectée dans raw_results.csv: utilisation des résultats agrégés."
        )
    if distribution_by_group:
        rows = [
            {
                "algo": algo,
                "snir_mode": snir_mode,
                **{f"sf{sf}_share": share for sf, share in values.items()},
            }
            for (algo, snir_mode), values in distribution_by_group.items()
        ]
    else:
        rows = filter_cluster(load_step1_aggregated(aggregated_results_path), "all")
    rows = [
        {
            **row,
            "algo": _normalize_algo(str(row.get("algo", ""))) or row.get("algo", ""),
            "snir_mode": _normalize_snir_mode(str(row.get("snir_mode", "")))
            or row.get("snir_mode", ""),
        }
        for row in rows
    ]

    fig = _plot_distribution(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S8", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
