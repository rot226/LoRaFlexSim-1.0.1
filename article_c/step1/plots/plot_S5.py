"""Trace la figure S5 (PDR par algorithme, SNIR on/off)."""

from __future__ import annotations

import csv
from pathlib import Path
from random import Random
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from article_c.common.plot_helpers import (
    ALGO_LABELS,
    SNIR_LABELS,
    SNIR_MODES,
    _sample_step1_rows,
    apply_plot_style,
    save_figure,
)

TARGET_NETWORK_SIZE = 1280
NETWORK_SIZE_COLUMNS = ("network_size", "density", "nodes", "num_nodes")
PDR_COLUMNS = ("pdr", "pdr_mean")
ALGO_COLUMNS = ("algo", "algorithm", "method")
SNIR_COLUMNS = ("snir_mode", "snir_state", "snir", "with_snir")
CLUSTER_COLUMNS = ("cluster",)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def _normalize_algo(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "adr": "adr",
        "mixra_h": "mixra_h",
        "mixra_hybrid": "mixra_h",
        "mixra_opt": "mixra_opt",
        "mixra_optimal": "mixra_opt",
        "mixraopt": "mixra_opt",
        "ucb1_sf": "ucb1_sf",
    }
    return aliases.get(normalized, normalized)


def _normalize_snir(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"snir_on", "on", "true", "1", "yes"}:
        return "snir_on"
    if lowered in {"snir_off", "off", "false", "0", "no"}:
        return "snir_off"
    if "on" in lowered:
        return "snir_on"
    if "off" in lowered:
        return "snir_off"
    return None


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _filtered_sample_rows() -> list[dict[str, object]]:
    rows = []
    for row in _sample_step1_rows():
        rows.append(
            {
                "density": row.get("density"),
                "algo": row.get("algo"),
                "snir_mode": row.get("snir_mode"),
                "cluster": row.get("cluster", "all"),
                "pdr": row.get("pdr_mean"),
            }
        )
    return rows


def _select_network_size(
    rows: list[dict[str, str]],
    size_col: str,
) -> tuple[int, list[dict[str, str]]]:
    sizes = sorted(
        {
            int(value)
            for row in rows
            if (value := _as_float(row.get(size_col))) is not None
        }
    )
    if not sizes:
        return TARGET_NETWORK_SIZE, []
    target = TARGET_NETWORK_SIZE if TARGET_NETWORK_SIZE in sizes else sizes[-1]
    filtered = [
        row
        for row in rows
        if (value := _as_float(row.get(size_col))) is not None and int(value) == target
    ]
    return target, filtered


def _extract_pdr_groups(
    rows: list[dict[str, str]],
) -> tuple[int, dict[tuple[str, str], list[float]]]:
    if not rows:
        return TARGET_NETWORK_SIZE, {}
    columns = rows[0].keys()
    size_col = _pick_column(columns, NETWORK_SIZE_COLUMNS)
    algo_col = _pick_column(columns, ALGO_COLUMNS)
    snir_col = _pick_column(columns, SNIR_COLUMNS)
    pdr_col = _pick_column(columns, PDR_COLUMNS)
    cluster_col = _pick_column(columns, CLUSTER_COLUMNS)
    if not size_col or not algo_col or not snir_col or not pdr_col:
        return TARGET_NETWORK_SIZE, {}

    target_size, filtered_rows = _select_network_size(rows, size_col)
    values_by_group: dict[tuple[str, str], list[float]] = {}
    for row in filtered_rows:
        if cluster_col and row.get(cluster_col) not in {"all", "", None}:
            continue
        algo = _normalize_algo(row.get(algo_col))
        snir_mode = _normalize_snir(row.get(snir_col))
        if algo is None or snir_mode not in SNIR_LABELS:
            continue
        pdr = _as_float(row.get(pdr_col))
        if pdr is None:
            continue
        values_by_group.setdefault((algo, snir_mode), []).append(pdr)
    return target_size, values_by_group


def _plot_pdr_distribution(
    values_by_group: dict[tuple[str, str], list[float]],
    *,
    target_size: int,
) -> plt.Figure:
    algorithms = [
        algo for algo in ALGO_LABELS.keys() if any(key[0] == algo for key in values_by_group)
    ]
    if not algorithms:
        algorithms = sorted({algo for algo, _ in values_by_group})

    fig, ax = plt.subplots()
    rng = Random(42)
    base_positions = list(range(len(algorithms)))
    offsets = {"snir_on": -0.18, "snir_off": 0.18}
    colors = {"snir_on": "#4c78a8", "snir_off": "#f58518"}

    for snir_mode in SNIR_MODES:
        data = [values_by_group.get((algo, snir_mode), []) for algo in algorithms]
        positions = [pos + offsets[snir_mode] for pos in base_positions]
        violins = ax.violinplot(
            data,
            positions=positions,
            widths=0.32,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violins["bodies"]:
            body.set_facecolor(colors[snir_mode])
            body.set_edgecolor("none")
            body.set_alpha(0.35)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": colors[snir_mode], "linewidth": 1.0},
            medianprops={"color": colors[snir_mode], "linewidth": 1.5},
            whiskerprops={"color": colors[snir_mode], "linewidth": 1.0},
            capprops={"color": colors[snir_mode], "linewidth": 1.0},
        )

        for pos, values in zip(positions, data, strict=False):
            for value in values:
                jitter = rng.uniform(-0.05, 0.05)
                ax.scatter(
                    pos + jitter,
                    value,
                    s=14,
                    color=colors[snir_mode],
                    alpha=0.55,
                    zorder=3,
                )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([ALGO_LABELS.get(algo, algo) for algo in algorithms])
    ax.set_xlabel("Algorithme")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title(f"Step 1 - Distribution du PDR (network size = {target_size})")
    ax.set_ylim(0.0, 1.05)

    legend_items = [
        Line2D([0], [0], color=colors[mode], lw=4, label=SNIR_LABELS[mode])
        for mode in SNIR_MODES
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    plt.subplots_adjust(top=0.82)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "raw_results.csv"
    raw_rows = _read_rows(results_path)
    if not raw_rows:
        sample_rows = _filtered_sample_rows()
        target_size = TARGET_NETWORK_SIZE
        values_by_group: dict[tuple[str, str], list[float]] = {}
        for row in sample_rows:
            if row.get("cluster") != "all":
                continue
            algo = row.get("algo")
            snir_mode = row.get("snir_mode")
            pdr_value = row.get("pdr")
            if isinstance(pdr_value, (int, float)) and algo and snir_mode in SNIR_LABELS:
                values_by_group.setdefault((str(algo), str(snir_mode)), []).append(
                    float(pdr_value)
                )
    else:
        target_size, values_by_group = _extract_pdr_groups(raw_rows)

    fig = _plot_pdr_distribution(values_by_group, target_size=target_size)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S5", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
