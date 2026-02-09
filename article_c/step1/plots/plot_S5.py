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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from article_c.common.plot_helpers import (
    BASE_DPI,
    MetricStatus,
    SNIR_LABELS,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    apply_output_fonttype,
    apply_figure_layout,
    assert_legend_present,
    clear_axis_legends,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    get_export_formats,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    render_metric_status,
    select_received_metric_key,
    save_figure,
    save_figure_path,
    warn_if_insufficient_network_sizes,
)
from article_c.common.plotting_style import SAVEFIG_STYLE
from article_c.common.utils import ensure_dir
from article_c.step1.plots.plot_utils import configure_figure
from plot_defaults import resolve_ieee_figsize

TARGET_NETWORK_SIZE = 1280
MAX_ROWS_PER_PAGE = 3
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

    boxplot_parts = ax.boxplot(
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
    for patch in boxplot_parts.get("boxes", []):
        patch.set_label(f"Boîte PDR ({SNIR_LABELS[snir_mode]})")

    rng = Random(42)
    jitter_x_range = 0.06
    jitter_y_range = 0.01
    max_points = 24
    for pos, values in zip(positions, data, strict=False):
        if not values:
            continue
        step = max(1, len(values) // max_points)
        label = f"Échantillons PDR ({SNIR_LABELS[snir_mode]})"
        label_added = False
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
                label=label if not label_added else None,
            )
            label_added = True

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])


def _plot_pdr_distributions(
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]],
    network_sizes: list[int],
    *,
    enable_suptitle: bool = True,
) -> list[plt.Figure]:
    legend_handles = [
        Patch(facecolor="#4c78a8", edgecolor="none", alpha=0.3, label=SNIR_LABELS["snir_on"]),
        Patch(facecolor="#f58518", edgecolor="none", alpha=0.3, label=SNIR_LABELS["snir_off"]),
    ]
    legend_labels = [handle.get_label() for handle in legend_handles]
    legend_ncol = 2
    legend_rows = max(1, math.ceil(len(legend_handles) / legend_ncol))
    all_values = [
        float(value)
        for groups in values_by_size.values()
        for values in groups.values()
        for value in values
        if isinstance(value, (int, float))
    ]
    metric_state = is_constant_metric(all_values)
    if metric_state is not MetricStatus.OK:
        fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(legend_handles)))
        render_metric_status(
            fig,
            ax,
            metric_state,
            show_fallback_legend=False,
            legend_handles=(legend_handles, legend_labels),
        )
        configure_axes = ax
        clear_axis_legends(configure_axes)
        final_loc, final_rows = configure_figure(
            fig,
            configure_axes,
            title=None,
            legend_loc="right",
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            enable_suptitle=enable_suptitle,
        )
        layout_margins = legend_margins(final_loc, legend_rows=final_rows)
        legend_bbox = None
        if fig.legends and final_loc == "above":
            legend_bbox = _legend_bbox(fig, final_rows)
            fig.legends[0].set_bbox_to_anchor(legend_bbox)
        apply_figure_layout(
            fig,
            margins=layout_margins,
            bbox_to_anchor=legend_bbox,
            legend_rows=final_rows,
        )
        return [fig]

    if not network_sizes:
        network_sizes = [TARGET_NETWORK_SIZE]
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

    row_specs: list[tuple[int, str, bool]] = [
        (size, algo, fallback)
        for size in network_sizes
        for (algo, fallback) in algorithms
    ]
    max_rows_per_page = max(1, MAX_ROWS_PER_PAGE)
    total_pages = max(1, math.ceil(len(row_specs) / max_rows_per_page))
    figures: list[plt.Figure] = []
    for page_index in range(total_pages):
        page_rows = row_specs[
            page_index * max_rows_per_page : (page_index + 1) * max_rows_per_page
        ]
        title_suffix = (
            f"page {page_index + 1}/{total_pages}" if total_pages > 1 else ""
        )
        fig = _plot_pdr_distribution_page(
            values_by_size,
            page_rows,
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            legend_rows=legend_rows,
            title_suffix=title_suffix,
            enable_suptitle=enable_suptitle,
        )
        figures.append(fig)
    return figures


def _plot_pdr_distribution_page(
    values_by_size: dict[int, dict[tuple[str, bool, str], list[float]]],
    row_specs: list[tuple[int, str, bool]],
    *,
    legend_handles: list[Patch],
    legend_labels: list[str],
    legend_rows: int,
    title_suffix: str,
    enable_suptitle: bool = True,
) -> plt.Figure:
    ncols = 2
    nrows = max(1, len(row_specs))
    base_width, base_height = resolve_ieee_figsize(ncols)
    height_per_row = base_height * 0.95
    figsize = (base_width, height_per_row * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    legend_bbox = _legend_bbox(fig, legend_rows)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for row_index, (size, algo, fallback) in enumerate(row_specs):
        values_by_group = values_by_size.get(size, {})
        for col_index, snir_mode in enumerate(SNIR_MODES):
            ax = axes[row_index][col_index]
            _plot_pdr_distribution(
                ax,
                values=values_by_group.get((algo, fallback, snir_mode), []),
                snir_mode=snir_mode,
            )
            ax.set_xlabel("SNIR mode")
            if col_index == 0:
                ax.set_ylabel(
                    f"{algo_label(algo, fallback)}\nPDR (prob.)",
                    fontsize=9,
                )
            else:
                ax.set_ylabel("PDR (prob.)", fontsize=9)
        if row_index == 0:
            axes[row_index][0].annotate(
                f"Taille réseau = {size} nœuds",
                xy=(0.02, 1.02),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                fontsize=7,
                color="#444444",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
            )

    base_title = "Figure S5 — PDR par algorithme et mode SNIR (tailles indiquées)"
    title = f"{base_title} ({title_suffix})" if title_suffix else base_title
    clear_axis_legends(axes)
    final_loc, final_rows = configure_figure(
        fig,
        axes,
        title=None,
        legend_loc="right",
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        enable_suptitle=enable_suptitle,
    )
    layout_margins = {
        **legend_margins(final_loc, legend_rows=final_rows),
        "hspace": 0.95,
        "wspace": 0.25,
    }
    if fig.legends and final_loc == "above":
        fig.legends[0].set_bbox_to_anchor(legend_bbox)
    apply_figure_layout(
        fig,
        margins=layout_margins,
        bbox_to_anchor=legend_bbox if final_loc == "above" else None,
        legend_rows=final_rows,
    )
    return fig


def _save_multipage_figures(
    figures: list[plt.Figure],
    output_dir: Path,
    stem: str,
) -> None:
    ensure_dir(output_dir)
    apply_output_fonttype()
    pdf_path = output_dir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            fig.savefig(pdf, format="pdf", dpi=BASE_DPI, **SAVEFIG_STYLE)
    extra_formats = [fmt for fmt in get_export_formats() if fmt != "pdf"]
    for index, fig in enumerate(figures, start=1):
        page_stem = f"{stem}_page{index}"
        for ext in extra_formats:
            save_figure_path(fig, output_dir / f"{page_stem}.{ext}", fmt=ext)


def _legend_bbox(fig: plt.Figure, legend_rows: int, anchor_x: float = 0.5) -> tuple[float, float]:
    fig_height = fig.get_size_inches()[1]
    base_y = 1.02
    row_offset = 0.01 * max(0, legend_rows - 1)
    height_adjust = max(0.0, min(0.02, (10.0 - fig_height) * 0.003))
    y_position = min(1.05, max(base_y, base_y + row_offset + height_adjust))
    return (anchor_x, y_position)


def _resolve_step1_intermediate_path(base_path: Path) -> Path | None:
    by_round = base_path.with_name("aggregated_results_by_round.csv")
    if by_round.exists():
        return by_round
    by_replication = base_path.with_name("aggregated_results_by_replication.csv")
    if by_replication.exists():
        return by_replication
    return None


def main(
    argv: list[str] | None = None,
    allow_sample: bool = True,
    enable_suptitle: bool = True,
) -> None:
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

    warn_if_insufficient_network_sizes(network_sizes)

    figures = _plot_pdr_distributions(
        values_by_size,
        network_sizes,
        enable_suptitle=enable_suptitle,
    )
    output_dir = step_dir / "plots" / "output"
    if len(figures) == 1:
        save_figure(figures[0], output_dir, "plot_S5", use_tight=False)
        assert_legend_present(figures[0], "plot_S5")
    else:
        _save_multipage_figures(figures, output_dir, "plot_S5")
        for index, fig in enumerate(figures, start=1):
            assert_legend_present(fig, f"plot_S5_page{index}")
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
