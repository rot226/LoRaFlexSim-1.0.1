"""Trace la figure S5 (PDR par algorithme, SNIR on/off)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from random import Random
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    load_step1_aggregated,
    save_figure,
)

TARGET_NETWORK_SIZE = 1280
NETWORK_SIZE_COLUMNS = ("network_size", "density", "nodes", "num_nodes")
PDR_COLUMNS = ("pdr",)
PDR_MEAN_COLUMNS = ("pdr_mean",)
PDR_STD_COLUMNS = ("pdr_std",)
PDR_COUNT_COLUMNS = ("pdr_count",)
RX_COLUMNS = ("rx_success", "rx", "rx_ok")
TX_COLUMNS = ("tx_total", "tx", "tx_attempts")
ALGO_COLUMNS = ("algo", "algorithm", "method")
SNIR_COLUMNS = ("snir_mode", "snir_state", "snir", "with_snir")
CLUSTER_COLUMNS = ("cluster",)
MIXRA_FALLBACK_COLUMNS = ("mixra_opt_fallback", "mixra_fallback", "fallback")


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


def _as_bool(value: str | None) -> bool:
    if value is None or value == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "vrai"}


def _available_network_sizes(rows: Iterable[dict[str, object]]) -> list[int]:
    sizes: set[int] = set()
    for row in rows:
        size_value = _as_float(row.get("network_size") or row.get("density"))
        if size_value is None:
            continue
        sizes.add(int(size_value))
    return sorted(sizes)


def _select_target_size(available_sizes: list[int], target_size: int) -> int:
    if not available_sizes:
        return target_size
    if target_size in available_sizes:
        return target_size
    closest = min(
        available_sizes,
        key=lambda size: (abs(size - target_size), -size),
    )
    warnings.warn(f"Target size not found, using size={closest}", stacklevel=2)
    return closest


def _extract_raw_pdr_groups(
    rows: list[dict[str, str]],
) -> dict[int, dict[tuple[str, bool, str], list[float]]]:
    if not rows:
        return {}
    columns = rows[0].keys()
    size_col = _pick_column(columns, NETWORK_SIZE_COLUMNS)
    algo_col = _pick_column(columns, ALGO_COLUMNS)
    snir_col = _pick_column(columns, SNIR_COLUMNS)
    pdr_col = _pick_column(columns, PDR_COLUMNS)
    rx_col = _pick_column(columns, RX_COLUMNS)
    tx_col = _pick_column(columns, TX_COLUMNS)
    cluster_col = _pick_column(columns, CLUSTER_COLUMNS)
    fallback_col = _pick_column(columns, MIXRA_FALLBACK_COLUMNS)
    if not size_col or not algo_col or not snir_col or (not pdr_col and not (rx_col and tx_col)):
        return {}

    has_cluster_values = False
    if cluster_col:
        has_cluster_values = any(
            row.get(cluster_col) not in {"all", "", None} for row in rows
        )

    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]] = {}
    for row in rows:
        if cluster_col and has_cluster_values and row.get(cluster_col) in {"all", "", None}:
            continue
        algo = _normalize_algo(row.get(algo_col))
        snir_mode = _normalize_snir(row.get(snir_col))
        if algo is None or snir_mode not in SNIR_LABELS:
            continue
        fallback = _as_bool(row.get(fallback_col)) if fallback_col else False
        if algo != "mixra_opt":
            fallback = False
        if algo == "mixra_opt" and fallback:
            continue
        pdr = _as_float(row.get(pdr_col)) if pdr_col else None
        if pdr is None and rx_col and tx_col:
            rx_value = _as_float(row.get(rx_col))
            tx_value = _as_float(row.get(tx_col))
            if rx_value is not None and tx_value and tx_value > 0:
                pdr = rx_value / tx_value
        if pdr is None:
            continue
        size_value = _as_float(row.get(size_col))
        if size_value is None:
            continue
        size = int(size_value)
        values_by_size.setdefault(size, {}).setdefault((algo, fallback, snir_mode), []).append(pdr)
    if values_by_size and not has_cluster_values:
        warnings.warn(
            "Aucune distribution par cluster détectée dans raw_results.csv; "
            "utilisation des valeurs agrégées cluster='all'.",
            stacklevel=2,
        )
    return values_by_size


def _sample_distribution(
    mean: float,
    std: float,
    count: int,
    rng: Random,
) -> list[float]:
    if count <= 1 or std <= 0:
        return [mean]
    values = [rng.gauss(mean, std) for _ in range(count)]
    return [min(1.0, max(0.0, value)) for value in values]


def _extract_aggregated_pdr_groups(
    rows: list[dict[str, object]],
) -> dict[int, dict[tuple[str, bool, str], list[float]]]:
    if not rows:
        return {}
    columns = rows[0].keys()
    size_col = _pick_column(columns, NETWORK_SIZE_COLUMNS)
    algo_col = _pick_column(columns, ALGO_COLUMNS)
    snir_col = _pick_column(columns, SNIR_COLUMNS)
    mean_col = _pick_column(columns, PDR_MEAN_COLUMNS)
    std_col = _pick_column(columns, PDR_STD_COLUMNS)
    count_col = _pick_column(columns, PDR_COUNT_COLUMNS)
    cluster_col = _pick_column(columns, CLUSTER_COLUMNS)
    fallback_col = _pick_column(columns, MIXRA_FALLBACK_COLUMNS)
    if not size_col or not algo_col or not snir_col or not mean_col:
        return {}

    rng = Random(42)
    has_cluster_values = False
    if cluster_col:
        has_cluster_values = any(
            row.get(cluster_col) not in {"all", "", None} for row in rows
        )

    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]] = {}
    for row in rows:
        if cluster_col and has_cluster_values and row.get(cluster_col) in {"all", "", None}:
            continue
        algo = _normalize_algo(row.get(algo_col))
        snir_mode = _normalize_snir(row.get(snir_col))
        if algo is None or snir_mode not in SNIR_LABELS:
            continue
        fallback = _as_bool(row.get(fallback_col)) if fallback_col else False
        if algo != "mixra_opt":
            fallback = False
        if algo == "mixra_opt" and fallback:
            continue
        mean_value = _as_float(row.get(mean_col))
        if mean_value is None:
            continue
        std_value = _as_float(row.get(std_col)) if std_col else 0.0
        count_value = _as_float(row.get(count_col)) if count_col else None
        count = int(count_value) if count_value and count_value > 0 else 1
        pdr_values = _sample_distribution(mean_value, std_value or 0.0, count, rng)
        size_value = _as_float(row.get(size_col))
        if size_value is None:
            continue
        size = int(size_value)
        values_by_size.setdefault(size, {}).setdefault((algo, fallback, snir_mode), []).extend(
            pdr_values
        )
    return values_by_size


def _plot_pdr_distribution(
    ax: plt.Axes,
    values_by_group: dict[tuple[str, bool, str], list[float]],
    *,
    target_size: int,
) -> None:
    if not any(values for values in values_by_group.values()):
        warnings.warn(
            f"Aucune donnée disponible pour size={target_size}, plot ignoré.",
            stacklevel=2,
        )
        return

    algorithms: list[tuple[str, bool]] = []
    for algo in ("adr", "mixra_h", "mixra_opt", "ucb1_sf"):
        for fallback in (False, True):
            if any(
                key[0] == algo
                and key[1] == fallback
                and values_by_group.get(key)
                for key in values_by_group
            ):
                algorithms.append((algo, fallback))
    if not algorithms:
        algorithms = sorted(
            {
                (algo, fallback)
                for (algo, fallback, _), values in values_by_group.items()
                if values
            }
        )

    rng = Random(42)
    base_positions = list(range(len(algorithms)))
    offsets = {"snir_on": -0.18, "snir_off": 0.18}
    colors = {"snir_on": "#4c78a8", "snir_off": "#f58518"}

    for snir_mode in SNIR_MODES:
        data_with_positions: list[tuple[float, list[float]]] = []
        for (algo, fallback), pos in zip(algorithms, base_positions, strict=False):
            values = values_by_group.get((algo, fallback, snir_mode), [])
            if not values:
                algo_name = f"{algo}{'_fallback' if fallback else ''}"
                warnings.warn(
                    f"No data for size={target_size}, algo={algo_name}, snir={snir_mode}.",
                    stacklevel=2,
                )
                continue
            data_with_positions.append((pos + offsets[snir_mode], values))

        if not data_with_positions:
            warnings.warn(
                f"Aucune donnée pour size={target_size} et snir={snir_mode}, trace ignorée.",
                stacklevel=2,
            )
            continue

        data = [values for _, values in data_with_positions]
        positions = [pos for pos, _ in data_with_positions]
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
    ax.set_xticklabels([algo_label(algo, fallback) for algo, fallback in algorithms])
    ax.set_xlabel("Algorithme")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title(f"Step 1 - Distribution du PDR (network size = {target_size})")
    ax.set_ylim(0.0, 1.05)


def _plot_pdr_distributions(
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]],
    network_sizes: list[int],
) -> plt.Figure:
    if not network_sizes:
        network_sizes = [TARGET_NETWORK_SIZE]
    n_sizes = len(network_sizes)
    ncols = 2 if n_sizes > 1 else 1
    nrows = math.ceil(n_sizes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = list(axes.ravel())

    for ax, size in zip(axes, network_sizes, strict=False):
        values_by_group = values_by_size.get(size, {})
        _plot_pdr_distribution(ax, values_by_group, target_size=size)

    for ax in axes[len(network_sizes) :]:
        ax.axis("off")

    legend_items = [
        Line2D([0], [0], color="#4c78a8", lw=4, label=SNIR_LABELS["snir_on"]),
        Line2D([0], [0], color="#f58518", lw=4, label=SNIR_LABELS["snir_off"]),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Step 1 - Distribution du PDR par taille de réseau")
    fig.subplots_adjust(top=0.80)
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
    raw_results_path = step_dir / "results" / "raw_results.csv"
    raw_rows = _read_rows(raw_results_path)
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]] = {}
    network_sizes: list[int] = []
    if raw_rows:
        ensure_network_size(raw_rows)
        raw_rows = filter_mixra_opt_fallback(raw_rows)
        if args.network_sizes:
            raw_rows, _ = filter_rows_by_network_sizes(raw_rows, args.network_sizes)
            network_sizes = sorted({int(row["network_size"]) for row in raw_rows})
        else:
            network_sizes = [
                _select_target_size(
                    _available_network_sizes(raw_rows),
                    TARGET_NETWORK_SIZE,
                )
            ]
        values_by_size = _extract_raw_pdr_groups(raw_rows)

    if not values_by_size:
        aggregated_path = step_dir / "results" / "aggregated_results.csv"
        aggregated_rows = load_step1_aggregated(aggregated_path, allow_sample=allow_sample)
        if not aggregated_rows and not allow_sample:
            warnings.warn(
                "CSV Step1 manquant ou vide, figure ignorée.",
                stacklevel=2,
            )
            return
        aggregated_rows = filter_mixra_opt_fallback(aggregated_rows)
        if args.network_sizes:
            aggregated_rows, _ = filter_rows_by_network_sizes(
                aggregated_rows,
                args.network_sizes,
            )
            network_sizes = sorted({int(row["network_size"]) for row in aggregated_rows})
        else:
            network_sizes = [
                _select_target_size(
                    _available_network_sizes(aggregated_rows),
                    TARGET_NETWORK_SIZE,
                )
            ]
        values_by_size = _extract_aggregated_pdr_groups(aggregated_rows)

    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)

    fig = _plot_pdr_distributions(values_by_size, network_sizes)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S5", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
