"""Trace la figure S5 (PDR par algorithme, SNIR on/off)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_MODES,
    SUPTITLE_Y,
    algo_label,
    apply_plot_style,
    apply_figure_layout,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    render_constant_metric,
    select_received_metric_key,
    save_figure,
)
from article_c.common.plotting_style import LEGEND_STYLE
from plot_defaults import DEFAULT_FIGSIZE_MULTI

TARGET_NETWORK_SIZE = 1280
NETWORK_SIZE_COLUMNS = ("network_size", "density", "nodes", "num_nodes")
PDR_COLUMNS = ("pdr",)
PDR_AGGREGATED_COLUMNS = ("aggregated_pdr",)
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
            "Aucune distribution par cluster détectée dans raw_metrics.csv; "
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
    aggregated_pdr_col = _pick_column(columns, PDR_AGGREGATED_COLUMNS)
    pdr_col = _pick_column(columns, PDR_COLUMNS)
    mean_col = _pick_column(columns, PDR_MEAN_COLUMNS)
    std_col = _pick_column(columns, PDR_STD_COLUMNS)
    count_col = _pick_column(columns, PDR_COUNT_COLUMNS)
    cluster_col = _pick_column(columns, CLUSTER_COLUMNS)
    fallback_col = _pick_column(columns, MIXRA_FALLBACK_COLUMNS)
    if not size_col or not algo_col or not snir_col or (
        not aggregated_pdr_col and not pdr_col and not mean_col
    ):
        return {}

    rng = Random(42)
    has_cluster_values = False
    if cluster_col:
        has_cluster_values = any(
            row.get(cluster_col) not in {"all", "", None} for row in rows
        )

    invalid_aggregated = 0
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
        pdr_values: list[float] = []
        if aggregated_pdr_col:
            aggregated_value = _as_float(row.get(aggregated_pdr_col))
            if aggregated_value and aggregated_value > 0:
                pdr_values = [aggregated_value]
            else:
                invalid_aggregated += 1
                continue
        elif pdr_col:
            pdr_value = _as_float(row.get(pdr_col))
            if pdr_value is None:
                continue
            pdr_values = [pdr_value]
        else:
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
    if aggregated_pdr_col and invalid_aggregated:
        warnings.warn(
            "Des lignes avec aggregated_pdr manquant ou nul ont été ignorées.",
            stacklevel=2,
        )
    return values_by_size


def _plot_pdr_distribution(
    ax: plt.Axes,
    *,
    values: list[float],
    snir_mode: str,
) -> None:
    color = "#4c78a8" if snir_mode == "snir_on" else "#f58518"
    if not values:
        ax.text(
            0.5,
            0.5,
            "Données manquantes",
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.0)
        return

    data = [values]
    positions = [0]
    violins = ax.violinplot(
        data,
        positions=positions,
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.1},
        medianprops={"color": color, "linewidth": 1.6},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
    )

    rng = Random(42)
    jitter_x_range = 0.06
    jitter_y_range = 0.01
    max_points = 24
    for pos, values in zip(positions, data, strict=False):
        if not values:
            continue
        step = max(1, len(values) // max_points)
        for value in values[::step]:
            jitter_x = rng.uniform(-jitter_x_range, jitter_x_range)
            jitter_y = rng.uniform(-jitter_y_range, jitter_y_range)
            jittered_value = min(1.0, max(0.0, value + jitter_y))
            ax.scatter(
                pos + jitter_x,
                jittered_value,
                s=16,
                color=color,
                alpha=0.6,
                zorder=3,
            )

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])


def _plot_pdr_distributions(
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]],
    network_sizes: list[int],
) -> plt.Figure:
    legend_handles = [
        Patch(facecolor="#4c78a8", edgecolor="none", alpha=0.3, label=SNIR_LABELS["snir_on"]),
        Patch(facecolor="#f58518", edgecolor="none", alpha=0.3, label=SNIR_LABELS["snir_off"]),
    ]
    legend_labels = [handle.get_label() for handle in legend_handles]
    legend_style = {**LEGEND_STYLE, "ncol": 2}
    legend_bbox = legend_style.get("bbox_to_anchor", (0.5, 1.02))
    all_values = [
        float(value)
        for groups in values_by_size.values()
        for values in groups.values()
        for value in values
        if isinstance(value, (int, float))
    ]
    if is_constant_metric(all_values):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_MULTI)
        apply_figure_layout(fig, figsize=(8, 5))
        render_constant_metric(
            fig,
            ax,
            show_fallback_legend=False,
            legend_handles=(legend_handles, legend_labels),
        )
        fig.legend(handles=legend_handles, **legend_style)
        fig.suptitle(
            "Figure S5 — PDR par algorithme et mode SNIR (tailles indiquées)",
            y=SUPTITLE_Y,
        )
        apply_figure_layout(fig, margins=legend_margins("above"), bbox_to_anchor=legend_bbox)
        return fig
    if not network_sizes:
        network_sizes = [TARGET_NETWORK_SIZE]
    n_sizes = len(network_sizes)
    algorithms: list[tuple[str, bool]] = []
    for algo in ("adr", "mixra_h", "mixra_opt", "ucb1_sf"):
        for fallback in (False, True):
            if any(
                key[0] == algo and key[1] == fallback
                for size in network_sizes
                for key in values_by_size.get(size, {})
            ):
                algorithms.append((algo, fallback))
    if not algorithms:
        algorithms = sorted(
            {
                (algo, fallback)
                for size in network_sizes
                for (algo, fallback, _), values in values_by_size.get(size, {}).items()
                if values
            }
        )

    ncols = 2
    nrows = max(1, n_sizes * max(1, len(algorithms)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 2.4 * nrows), sharey=True)
    apply_figure_layout(fig, figsize=(6.2 * ncols, 2.4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for size_index, size in enumerate(network_sizes):
        values_by_group = values_by_size.get(size, {})
        for algo_index, (algo, fallback) in enumerate(algorithms):
            row_index = size_index * len(algorithms) + algo_index
            for col_index, snir_mode in enumerate(SNIR_MODES):
                ax = axes[row_index][col_index]
                _plot_pdr_distribution(
                    ax,
                    values=values_by_group.get((algo, fallback, snir_mode), []),
                    snir_mode=snir_mode,
                )
                if col_index == 0:
                    ax.set_ylabel(
                        f"{algo_label(algo, fallback)}\nPDR (ratio 0–1)",
                        fontsize=9,
                    )
                if row_index != nrows - 1:
                    ax.set_xlabel("")
            if size_index == 0 and algo_index == 0:
                axes[row_index][0].set_title(SNIR_LABELS["snir_on"])
                axes[row_index][1].set_title(SNIR_LABELS["snir_off"])
            axes[row_index][0].annotate(
                f"Taille réseau = {size} nœuds",
                xy=(0.02, 1.02),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                fontsize=8,
                color="#444444",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
            )

    fig.legend(handles=legend_handles, **legend_style)
    fig.suptitle(
        "Figure S5 — PDR par algorithme et mode SNIR (tailles indiquées)",
        y=SUPTITLE_Y,
    )
    apply_figure_layout(
        fig,
        margins={
            **legend_margins("above"),
            "hspace": 0.75,
            "wspace": 0.25,
        },
        bbox_to_anchor=legend_bbox,
    )
    return fig


def _resolve_step1_intermediate_path(base_path: Path) -> Path | None:
    by_round = base_path.with_name("aggregated_results_by_round.csv")
    if by_round.exists():
        return by_round
    by_replication = base_path.with_name("aggregated_results_by_replication.csv")
    if by_replication.exists():
        return by_replication
    return None


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
    raw_results_path = step_dir / "results" / "raw_metrics.csv"
    raw_rows = _read_rows(raw_results_path)
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]] = {}
    network_sizes: list[int] = []
    if raw_rows:
        ensure_network_size(raw_rows)
        raw_rows = filter_mixra_opt_fallback(raw_rows)
        select_received_metric_key(raw_rows, "received_mean")
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
        intermediate_path = _resolve_step1_intermediate_path(aggregated_path)
        aggregated_source = intermediate_path or aggregated_path
        aggregated_rows = load_step1_aggregated(aggregated_source, allow_sample=allow_sample)
        if not aggregated_rows and not allow_sample:
            warnings.warn(
                "CSV Step1 manquant ou vide, figure ignorée.",
                stacklevel=2,
            )
            return
        aggregated_rows = filter_mixra_opt_fallback(aggregated_rows)
        select_received_metric_key(aggregated_rows, "received_mean")
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
