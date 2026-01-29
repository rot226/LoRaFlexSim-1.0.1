"""Utilitaires de traçage pour les figures de l'article C."""

from __future__ import annotations

import csv
import logging
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from article_c.common.plotting_style import (
    LEGEND_STYLE,
    SAVEFIG_STYLE,
    set_network_size_ticks,
)
from article_c.common.utils import ensure_dir
from article_c.common.config import DEFAULT_CONFIG

ALGO_LABELS = {
    "adr": "ADR",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
    "ucb1_sf": "UCB1-SF",
}
ALGO_COLORS = {
    "adr": "#1f77b4",
    "mixra_h": "#ff7f0e",
    "mixra_opt": "#2ca02c",
    "ucb1_sf": "#d62728",
}
ALGO_MARKERS = {
    "adr": "o",
    "mixra_h": "s",
    "mixra_opt": "^",
    "ucb1_sf": "D",
}
SNIR_MODES = ("snir_on", "snir_off")
SNIR_LABELS = {
    "snir_on": "SNIR on",
    "snir_off": "SNIR off",
}
SNIR_LINESTYLES = {
    "snir_on": "solid",
    "snir_off": "dashed",
}
MIXRA_FALLBACK_COLUMNS = ("mixra_opt_fallback", "mixra_fallback", "fallback")
LOGGER = logging.getLogger(__name__)
DERIVED_SUFFIXES = ("_mean", "_std", "_count", "_ci95", "_p10", "_p50", "_p90")
RECEIVED_MEAN_KEY = "received_mean"
RECEIVED_ALGO_MEAN_KEY = "received_algo_mean"
RECEIVED_ALGO_TOL = 1e-6
BASE_FIGSIZE = (7.2, 4.2)
BASE_FONT_FAMILY = "sans-serif"
BASE_FONT_SANS = ["DejaVu Sans", "Arial", "Liberation Sans"]
BASE_FONT_SIZE = 10
BASE_LINE_WIDTH = 1.6
BASE_GRID_COLOR = "#e0e0e0"
BASE_GRID_ALPHA = 0.6
BASE_GRID_LINEWIDTH = 0.8
BASE_DPI = 300
BASE_GRID_ENABLED = True
AXES_TITLE_Y = 1.02
SUPTITLE_Y = 0.965
FIGURE_SUBPLOT_TOP = 0.78
LEGEND_TOP_MARGIN = 0.74
LEGEND_TOP_RESERVED = 0.02
LEGEND_ABOVE_TIGHT_LAYOUT_TOP = 0.86
LEGEND_RIGHT_MARGIN = 0.78
CONSTANT_METRIC_VARIANCE_THRESHOLD = 1e-6
CONSTANT_METRIC_MESSAGE = "métrique constante – à investiguer"


def apply_plot_style() -> None:
    """Applique un style homogène pour les figures Step1/Step2."""
    plt.rcParams.update(
        {
            "figure.figsize": BASE_FIGSIZE,
            "figure.subplot.top": FIGURE_SUBPLOT_TOP,
            "figure.dpi": BASE_DPI,
            "axes.grid": BASE_GRID_ENABLED,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titley": AXES_TITLE_Y,
            "grid.color": BASE_GRID_COLOR,
            "grid.alpha": BASE_GRID_ALPHA,
            "grid.linewidth": BASE_GRID_LINEWIDTH,
            "font.family": BASE_FONT_FAMILY,
            "font.sans-serif": BASE_FONT_SANS,
            "font.size": BASE_FONT_SIZE,
            "lines.linewidth": BASE_LINE_WIDTH,
            "savefig.dpi": BASE_DPI,
        }
    )
    plt.subplots_adjust(top=FIGURE_SUBPLOT_TOP)


def _flatten_axes(axes: object) -> list[plt.Axes]:
    if isinstance(axes, plt.Axes):
        return [axes]
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        flattened: list[plt.Axes] = []
        for item in axes:
            flattened.extend(_flatten_axes(item))
        return flattened
    return []


def render_constant_metric(
    fig: plt.Figure,
    axes: object,
    *,
    message: str = CONSTANT_METRIC_MESSAGE,
    legend_loc: str = "above",
    show_fallback_legend: bool = True,
    legend_handles: tuple[list[Line2D], list[str]] | None = None,
) -> None:
    """Affiche un message centré lorsque la métrique est constante.

    legend_handles permet de fournir des handles/labels factices pour la légende.
    """
    for ax in _flatten_axes(axes):
        ax.clear()
        ax.axis("off")
    fig.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color="#444444",
    )
    if show_fallback_legend and not _figure_has_legend(fig):
        if legend_handles is None:
            handles, labels = fallback_legend_handles()
        else:
            handles, labels = legend_handles
        if handles:
            legend_style = _legend_style(legend_loc, len(labels))
            fig.legend(handles, labels, **legend_style)
            apply_figure_layout(
                fig,
                margins=_legend_margins(legend_loc),
                bbox_to_anchor=legend_style.get("bbox_to_anchor"),
            )


def _metric_variance(
    values: list[float],
) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)


def is_constant_metric(
    values: list[float],
    *,
    threshold: float = CONSTANT_METRIC_VARIANCE_THRESHOLD,
) -> bool:
    if not values:
        return True
    return _metric_variance(values) < threshold


def metric_values(
    rows: list[dict[str, object]],
    metric_key: str,
) -> list[float]:
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    values: list[float] = []
    for row in rows:
        value = row.get(median_key)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return values


def select_received_metric_key(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    derived_key: str = RECEIVED_ALGO_MEAN_KEY,
    tolerance: float = RECEIVED_ALGO_TOL,
) -> str:
    """Retourne la clé de métrique à utiliser pour les métriques de réception.

    Source officielle: la valeur "received" est dérivée de sent_mean * pdr_mean.
    On calcule donc received_algo_mean à partir de ces champs pour éviter toute
    divergence future entre valeurs agrégées et dérivées par algorithme.
    """
    if metric_key != RECEIVED_MEAN_KEY:
        return metric_key
    differences: list[float] = []
    for row in rows:
        sent = row.get("sent_mean")
        pdr = row.get("pdr_mean")
        if isinstance(sent, (int, float)) and isinstance(pdr, (int, float)):
            derived_value = sent * pdr
            row[derived_key] = derived_value
            received = row.get(metric_key)
            if isinstance(received, (int, float)):
                differences.append(abs(received - derived_value))
        else:
            row[derived_key] = row.get(metric_key, 0.0)
    if differences and max(differences) > tolerance:
        warnings.warn(
            "received_mean ne correspond pas partout à sent_mean*pdr_mean; "
            "utilisation de received_algo_mean dérivé.",
            stacklevel=2,
        )
    return derived_key


def place_legend(ax: plt.Axes, *, legend_loc: str = "above") -> None:
    """Place la légende selon les consignes (au-dessus ou à droite)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        handles, labels = fallback_legend_handles()
    if not handles:
        return
    legend_style = _legend_style(legend_loc, len(labels))
    ax.figure.legend(handles, labels, **legend_style)
    apply_figure_layout(
        ax.figure,
        margins=_legend_margins(legend_loc),
        bbox_to_anchor=legend_style.get("bbox_to_anchor"),
    )


def _legend_style(legend_loc: str, label_count: int | None = None) -> dict[str, object]:
    normalized = _normalize_legend_loc(legend_loc)
    if normalized == "right":
        return {
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "ncol": 1,
            "frameon": False,
        }
    legend_style = dict(LEGEND_STYLE)
    if label_count is not None and "ncol" in legend_style:
        legend_style["ncol"] = min(label_count, int(legend_style["ncol"]))
    return legend_style


def _legend_margins(legend_loc: str) -> dict[str, float]:
    normalized = _normalize_legend_loc(legend_loc)
    if normalized == "right":
        return {"top": FIGURE_SUBPLOT_TOP, "right": LEGEND_RIGHT_MARGIN}
    if normalized == "above":
        return {"top": LEGEND_TOP_MARGIN}
    return {"top": FIGURE_SUBPLOT_TOP}


def legend_margins(legend_loc: str) -> dict[str, float]:
    """Expose les marges recommandées pour une légende donnée."""
    return _legend_margins(legend_loc)


def _normalize_legend_loc(legend_loc: str) -> str:
    normalized = str(legend_loc or "").strip().lower()
    if normalized in {"haut", "top", "above"}:
        return "above"
    if normalized in {"droite", "right"}:
        return "right"
    return normalized


def _figure_has_legend(fig: plt.Figure) -> bool:
    if fig.legends:
        return True
    return any(ax.get_legend() is not None for ax in fig.axes)


def legend_handles_for_algos_snir(
    snir_modes: Iterable[str] | None = None,
) -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    normalized_snir_modes = [
        str(mode).strip().lower() for mode in (snir_modes or SNIR_MODES)
    ]
    for algo_key, algo_label_value in ALGO_LABELS.items():
        color = ALGO_COLORS.get(algo_key, "#333333")
        marker = ALGO_MARKERS.get(algo_key, "o")
        for snir_mode in normalized_snir_modes:
            if snir_mode not in SNIR_LINESTYLES or snir_mode not in SNIR_LABELS:
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle=SNIR_LINESTYLES[snir_mode],
                    linewidth=BASE_LINE_WIDTH,
                    markersize=5.5,
                )
            )
            labels.append(f"{algo_label_value} ({SNIR_LABELS[snir_mode]})")
    return handles, labels


def fallback_legend_handles() -> tuple[list[Line2D], list[str]]:
    return legend_handles_for_algos_snir()


def add_global_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    legend_loc: str = "above",
    handles: list[Line2D] | None = None,
    labels: list[str] | None = None,
    use_fallback: bool = True,
) -> None:
    """Ajoute une légende globale à la figure."""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles and use_fallback:
        handles, labels = fallback_legend_handles()
    if not handles:
        return
    legend_style = _legend_style(legend_loc, len(labels))
    fig.legend(handles, labels, **legend_style)
    ncol = int(legend_style.get("ncol", len(labels)) or 1)
    legend_rows = max(1, math.ceil(len(labels) / ncol))
    apply_figure_layout(
        fig,
        margins=_legend_margins(legend_loc),
        bbox_to_anchor=legend_style.get("bbox_to_anchor"),
        legend_rows=legend_rows,
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    use_tight: bool = False,
) -> None:
    """Sauvegarde la figure en PNG et PDF dans le répertoire cible.

    Sur Windows, privilégier bbox_inches=None (valeur par défaut).
    """
    ensure_dir(output_dir)
    if use_tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=BASE_DPI, **SAVEFIG_STYLE)


def apply_figure_layout(
    fig: plt.Figure,
    *,
    figsize: tuple[float, float] | None = None,
    tight_layout: bool | Mapping[str, object] = False,
    bbox_to_anchor: tuple[float, float] | None = None,
    margins: dict[str, float] | None = None,
    legend_rows: int = 1,
) -> None:
    """Applique taille, marges, légendes et tight_layout sur une figure."""
    layout_rect: tuple[float, float, float, float] | None = None
    extra_legend_rows = max(0, legend_rows - 1)
    reserved_top = 0.0
    if figsize is not None:
        fig.set_size_inches(*figsize, forward=True)
    if margins is None and fig.legends:
        margins = {"top": LEGEND_TOP_MARGIN}
    if margins:
        adjusted_margins = dict(margins)
        if extra_legend_rows and "top" in adjusted_margins:
            adjusted_margins["top"] = max(
                0.0, adjusted_margins["top"] - 0.05 * extra_legend_rows
            )
        if "top" in adjusted_margins:
            reserved_top = min(LEGEND_TOP_RESERVED, adjusted_margins["top"])
        fig.subplots_adjust(**adjusted_margins)
        layout_rect = (
            adjusted_margins.get("left", 0.0),
            adjusted_margins.get("bottom", 0.0),
            adjusted_margins.get("right", 1.0),
            max(0.0, adjusted_margins.get("top", 1.0) - reserved_top),
        )
    if bbox_to_anchor is not None:
        legends = list(fig.legends)
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                legends.append(legend)
        for legend in legends:
            legend.set_bbox_to_anchor(bbox_to_anchor)
    if tight_layout:
        if isinstance(tight_layout, Mapping):
            adjusted_tight = dict(tight_layout)
            rect = adjusted_tight.get("rect")
            if extra_legend_rows and rect:
                left, bottom, right, top = rect
                adjusted_tight["rect"] = (
                    left,
                    bottom,
                    right,
                    max(0.0, top - 0.05 * extra_legend_rows),
                )
            if reserved_top and adjusted_tight.get("rect"):
                left, bottom, right, top = adjusted_tight["rect"]
                adjusted_tight["rect"] = (
                    left,
                    bottom,
                    right,
                    max(0.0, top - reserved_top),
                )
            fig.tight_layout(**adjusted_tight)
        else:
            fig.tight_layout()
    elif layout_rect is not None:
        fig.tight_layout(rect=layout_rect)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "vrai"}


def _normalize_algo(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_mixra_opt(row: dict[str, object]) -> bool:
    return _normalize_algo(row.get("algo")) == "mixra_opt"


def _mixra_opt_fallback(row: dict[str, object]) -> bool:
    for key in MIXRA_FALLBACK_COLUMNS:
        if key in row:
            return _to_bool(row.get(key))
    return False


def filter_mixra_opt_fallback(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    has_mixra_opt = any(_is_mixra_opt(row) for row in rows)
    filtered = [
        row
        for row in rows
        if not (_is_mixra_opt(row) and _mixra_opt_fallback(row))
    ]
    has_valid_mixra_opt = any(_is_mixra_opt(row) for row in filtered)
    if has_mixra_opt and not has_valid_mixra_opt:
        warnings.warn("MixRA-Opt absent (fallback)", stacklevel=2)
    return filtered


def load_step1_aggregated(
    path: Path,
    *,
    allow_sample: bool = False,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV vide pour Step1: {path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        network_size_value = row.get("density")
        if "network_size" in row and row.get("network_size") not in (None, ""):
            network_size_value = row.get("network_size")
        network_size = _to_float(network_size_value)
        parsed_row: dict[str, object] = {
            "network_size": network_size,
            "algo": row.get("algo", ""),
            "snir_mode": row.get("snir_mode", ""),
            "cluster": row.get("cluster", "all"),
            "mixra_opt_fallback": _to_bool(row.get("mixra_opt_fallback")),
        }
        if "density" in row:
            parsed_row["density"] = _to_float(row.get("density"))
        for key, value in row.items():
            if key in {
                "density",
                "network_size",
                "algo",
                "snir_mode",
                "cluster",
                "mixra_opt_fallback",
            }:
                continue
            parsed_row[key] = _to_float(value)
        parsed.append(parsed_row)
    return parsed


def load_step2_aggregated(
    path: Path,
    *,
    allow_sample: bool = False,
) -> list[dict[str, object]]:
    intermediate_path = _resolve_intermediate_step2_path(path)
    source_path = intermediate_path or path
    rows = _read_csv_rows(source_path)
    if not rows:
        raise ValueError(f"CSV vide pour Step2: {source_path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        network_size_value = row.get("density")
        if "network_size" in row and row.get("network_size") not in (None, ""):
            network_size_value = row.get("network_size")
        network_size = _to_float(network_size_value)
        parsed_row: dict[str, object] = {
            "network_size": network_size,
            "algo": row.get("algo", ""),
            "snir_mode": row.get("snir_mode", ""),
            "cluster": row.get("cluster", "all"),
        }
        if "mixra_opt_fallback" in row:
            parsed_row["mixra_opt_fallback"] = row.get("mixra_opt_fallback")
        if "density" in row:
            parsed_row["density"] = _to_float(row.get("density"))
        for key, value in row.items():
            if key in {
                "density",
                "network_size",
                "algo",
                "snir_mode",
                "cluster",
                "mixra_opt_fallback",
            }:
                continue
            parsed_row[key] = _to_float(value)
        parsed.append(parsed_row)
    if intermediate_path is None:
        return parsed
    return _aggregate_step2_intermediate(parsed)


def _resolve_intermediate_step2_path(path: Path) -> Path | None:
    by_round = path.with_name("aggregated_results_by_round.csv")
    if by_round.exists():
        return by_round
    by_replication = path.with_name("aggregated_results_by_replication.csv")
    if by_replication.exists():
        return by_replication
    return None


def _aggregate_step2_intermediate(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    group_keys = ["network_size", "algo", "snir_mode", "cluster"]
    if any(row.get("mixra_opt_fallback") not in (None, "") for row in rows):
        group_keys.append("mixra_opt_fallback")
    group_keys_tuple = tuple(group_keys)
    numeric_keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in group_keys_tuple or key == "density":
                continue
            if any(key.endswith(suffix) for suffix in DERIVED_SUFFIXES):
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group_key = tuple(row.get(key) for key in group_keys_tuple)
        groups.setdefault(group_key, []).append(row)
    aggregated: list[dict[str, object]] = []
    for group_key, grouped_rows in groups.items():
        aggregated_row: dict[str, object] = dict(zip(group_keys_tuple, group_key))
        for key in sorted(numeric_keys):
            values = [
                row[key]
                for row in grouped_rows
                if isinstance(row.get(key), (int, float))
            ]
            count = len(values)
            if values:
                mean_value = sum(values) / count
                if count > 1:
                    variance = sum((value - mean_value) ** 2 for value in values) / (
                        count - 1
                    )
                    std_value = math.sqrt(variance)
                else:
                    std_value = 0.0
            else:
                mean_value = 0.0
                std_value = 0.0
            ci95_value = 1.96 * std_value / math.sqrt(count) if count > 1 else 0.0
            aggregated_row[f"{key}_mean"] = mean_value
            aggregated_row[f"{key}_std"] = std_value
            aggregated_row[f"{key}_count"] = count
            aggregated_row[f"{key}_ci95"] = ci95_value
            sorted_values = sorted(values)
            aggregated_row[f"{key}_p10"] = _percentile(sorted_values, 10)
            aggregated_row[f"{key}_p50"] = _percentile(sorted_values, 50)
            aggregated_row[f"{key}_p90"] = _percentile(sorted_values, 90)
        aggregated.append(aggregated_row)
    return aggregated


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower]) + (float(values[upper]) - float(values[lower])) * weight


def load_step2_selection_probs(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV vide pour Step2 (sélections): {path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {
            "round": int(_to_float(row.get("round"))),
            "sf": int(_to_float(row.get("sf"))),
            "selection_prob": _to_float(row.get("selection_prob")),
        }
        if "network_size" in row and row.get("network_size") not in (None, ""):
            parsed_row["network_size"] = _to_float(row.get("network_size"))
        parsed.append(parsed_row)
    return parsed


def algo_labels(algorithms: Iterable[object]) -> list[str]:
    labels: list[str] = []
    for algo in algorithms:
        if isinstance(algo, tuple) and len(algo) == 2:
            label = algo_label(str(algo[0]), bool(algo[1]))
        else:
            label = algo_label(str(algo))
        labels.append(label)
    return labels


def algo_label(algo: str, fallback: bool = False) -> str:
    canonical = _normalize_algo(algo)
    return ALGO_LABELS.get(canonical, algo)


def filter_cluster(rows: list[dict[str, object]], cluster: str) -> list[dict[str, object]]:
    if any("cluster" in row for row in rows):
        return [row for row in rows if row.get("cluster") == cluster]
    return rows


def ensure_network_size(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row.get("network_size") in (None, "") and "density" in row:
            row["network_size"] = row["density"]


def normalize_network_size_rows(rows: list[dict[str, object]]) -> None:
    ensure_network_size(rows)
    for row in rows:
        row["network_size"] = int(_to_float(row.get("network_size")))


def warn_if_missing_network_sizes(
    requested: Iterable[int] | None,
    available: Iterable[int],
) -> None:
    if not requested:
        return
    requested_sizes = sorted({int(_to_float(size)) for size in requested})
    available_sizes = sorted({int(_to_float(size)) for size in available})
    missing = sorted(set(requested_sizes) - set(available_sizes))
    if missing:
        warnings.warn(
            "Tailles de réseau absentes: "
            + ", ".join(str(size) for size in missing)
            + ". Tailles disponibles: "
            + ", ".join(str(size) for size in available_sizes),
            stacklevel=2,
        )


def _network_size_value(row: dict[str, object]) -> int:
    if "network_size" in row:
        return int(_to_float(row.get("network_size")))
    return int(_to_float(row.get("density")))


def filter_rows_by_network_sizes(
    rows: list[dict[str, object]],
    network_sizes: Iterable[int] | None,
) -> tuple[list[dict[str, object]], list[int]]:
    normalize_network_size_rows(rows)
    unique_network_sizes = sorted(
        {row["network_size"] for row in rows if "network_size" in row}
    )
    LOGGER.info("network_size uniques après conversion: %s", unique_network_sizes)
    available = sorted({_network_size_value(row) for row in rows})
    if not network_sizes:
        return rows, available
    requested = sorted({int(_to_float(size)) for size in network_sizes})
    warn_if_missing_network_sizes(requested, available)
    filtered = [row for row in rows if row["network_size"] in requested]
    if not filtered:
        warnings.warn(
            "Aucune taille de réseau trouvée. Tailles disponibles: "
            + ", ".join(str(size) for size in available),
            stacklevel=2,
        )
    return filtered, available


def plot_metric_by_snir(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    use_algo_styles: bool = False,
    algo_colors: dict[str, str] | None = None,
    algo_markers: dict[str, str] | None = None,
    line_width: float = 1.6,
    marker_size: float = 5.5,
    percentile_line_width: float = 1.1,
) -> None:
    network_sizes = sorted({_network_size_value(row) for row in rows})
    median_key, lower_key, upper_key = resolve_percentile_keys(rows, metric_key)

    def _algo_key(row: dict[str, object]) -> tuple[str, bool]:
        algo_value = str(row.get("algo", ""))
        fallback = bool(row.get("mixra_opt_fallback")) if _is_mixra_opt(row) else False
        return algo_value, fallback

    algorithms = sorted({_algo_key(row) for row in rows})
    for algo, fallback in algorithms:
        algo_rows = [row for row in rows if _algo_key(row) == (algo, fallback)]
        densities = sorted({_network_size_value(row) for row in algo_rows})
        normalized_algo = _normalize_algo(algo)
        color = None
        marker = "o"
        if use_algo_styles:
            color = (algo_colors or ALGO_COLORS).get(normalized_algo)
            marker = (algo_markers or ALGO_MARKERS).get(normalized_algo, "o")
        for snir_mode in SNIR_MODES:
            points = {
                _network_size_value(row): row.get(median_key)
                for row in algo_rows
                if row["snir_mode"] == snir_mode
            }
            if not points:
                continue
            values = [
                _value_or_nan(points.get(density, float("nan"))) for density in densities
            ]
            label = f"{algo_label(algo, fallback)} ({SNIR_LABELS[snir_mode]})"
            line = ax.plot(
                densities,
                values,
                color=color,
                marker=marker,
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
                linewidth=line_width,
                markersize=marker_size,
            )[0]
            if lower_key and upper_key:
                lower_points = {
                    _network_size_value(row): row.get(lower_key)
                    for row in algo_rows
                    if row["snir_mode"] == snir_mode
                }
                upper_points = {
                    _network_size_value(row): row.get(upper_key)
                    for row in algo_rows
                    if row["snir_mode"] == snir_mode
                }
                lower_values = [
                    _value_or_nan(lower_points.get(density, float("nan")))
                    for density in densities
                ]
                upper_values = [
                    _value_or_nan(upper_points.get(density, float("nan")))
                    for density in densities
                ]
                color = line.get_color()
                ax.plot(
                    densities,
                    lower_values,
                    linestyle=":",
                    color=color,
                    alpha=0.6,
                    linewidth=percentile_line_width,
                )
                ax.plot(
                    densities,
                    upper_values,
                    linestyle=":",
                    color=color,
                    alpha=0.6,
                    linewidth=percentile_line_width,
                )
    set_network_size_ticks(ax, network_sizes)


def resolve_percentile_keys(
    rows: list[dict[str, object]],
    metric_key: str,
) -> tuple[str, str | None, str | None]:
    median_key = metric_key
    lower_key = None
    upper_key = None
    if metric_key.endswith("_mean"):
        base_key = metric_key[: -len("_mean")]
        p10_key = f"{base_key}_p10"
        p50_key = f"{base_key}_p50"
        p90_key = f"{base_key}_p90"
        if any(p50_key in row for row in rows):
            median_key = p50_key
        if any(p10_key in row for row in rows) and any(p90_key in row for row in rows):
            lower_key = p10_key
            upper_key = p90_key
    return median_key, lower_key, upper_key


def plot_metric_by_algo(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int],
    *,
    label_fn: Callable[[object], str] | None = None,
) -> None:
    median_key, lower_key, upper_key = resolve_percentile_keys(rows, metric_key)
    algorithms = sorted({row.get("algo") for row in rows})
    single_size = len(network_sizes) == 1
    only_size = network_sizes[0] if single_size else None
    label_fn = label_fn or (lambda algo: algo_label(str(algo)))
    for algo in algorithms:
        points = {
            int(row["network_size"]): row.get(median_key)
            for row in rows
            if row.get("algo") == algo
        }
        if single_size:
            value = points.get(only_size)
            if _is_invalid_value(value):
                continue
            if lower_key and upper_key:
                low = _value_or_nan(
                    next(
                        (
                            row.get(lower_key)
                            for row in rows
                            if row.get("algo") == algo
                            and int(row["network_size"]) == only_size
                        ),
                        float("nan"),
                    )
                )
                high = _value_or_nan(
                    next(
                        (
                            row.get(upper_key)
                            for row in rows
                            if row.get("algo") == algo
                            and int(row["network_size"]) == only_size
                        ),
                        float("nan"),
                    )
                )
                if not _is_invalid_value(low) and not _is_invalid_value(high):
                    yerr = [[value - low], [high - value]]
                    ax.errorbar([only_size], [value], yerr=yerr, fmt="o", label=label_fn(algo))
                    continue
            ax.scatter([only_size], [value], label=label_fn(algo))
            continue
        values = [_value_or_nan(points.get(size, float("nan"))) for size in network_sizes]
        line = ax.plot(network_sizes, values, marker="o", label=label_fn(algo))[0]
        if lower_key and upper_key:
            lower_points = {
                int(row["network_size"]): row.get(lower_key)
                for row in rows
                if row.get("algo") == algo
            }
            upper_points = {
                int(row["network_size"]): row.get(upper_key)
                for row in rows
                if row.get("algo") == algo
            }
            lower_values = [
                _value_or_nan(lower_points.get(size, float("nan")))
                for size in network_sizes
            ]
            upper_values = [
                _value_or_nan(upper_points.get(size, float("nan")))
                for size in network_sizes
            ]
            color = line.get_color()
            ax.plot(network_sizes, lower_values, linestyle=":", color=color, alpha=0.6)
            ax.plot(network_sizes, upper_values, linestyle=":", color=color, alpha=0.6)


def _is_invalid_value(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _value_or_nan(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _sample_step1_rows() -> list[dict[str, object]]:
    densities = [50, 100, 150]
    algos = ["adr", "mixra_h", "mixra_opt"]
    clusters = list(DEFAULT_CONFIG.qos.clusters) + ["all"]
    sf_values = list(DEFAULT_CONFIG.radio.spreading_factors)
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for algo in algos:
            for idx, density in enumerate(densities):
                for cluster in clusters:
                    base = 0.9 - 0.1 * idx
                    penalty = 0.05 if snir_mode == "snir_off" else 0.0
                    modifier = 0.02 * algos.index(algo)
                    cluster_bonus = 0.02 if cluster == "gold" else 0.0
                    pdr = max(
                        0.0, min(1.0, base - penalty + modifier + cluster_bonus)
                    )
                    pdr_std = 0.01 + 0.005 * idx + 0.002 * algos.index(algo)
                    pdr_ci95 = 1.96 * pdr_std / 5**0.5
                    algo_idx = algos.index(algo)
                    weights = []
                    for sf in sf_values:
                        sf_idx = sf_values.index(sf)
                        bias = 1.2 - 0.15 * sf_idx + 0.05 * algo_idx
                        if snir_mode == "snir_off":
                            bias += 0.05 * sf_idx
                        weights.append(max(0.05, bias))
                    total_weight = sum(weights) or 1.0
                    sf_shares = [weight / total_weight for weight in weights]
                    base_toa = 40.0 + 8.0 * idx + 4.0 * algo_idx
                    if snir_mode == "snir_off":
                        base_toa += 5.0
                    mean_toa_s = (base_toa + 0.2 * density) / 1000.0
                    row = {
                        "density": density,  # Alias legacy de network_size.
                        "network_size": density,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": cluster,
                        "mixra_opt_fallback": False,
                        "pdr_mean": pdr,
                        "pdr_std": pdr_std,
                        "pdr_ci95": pdr_ci95,
                        "sent_mean": 120 * density,
                        "received_mean": 120 * density * pdr,
                        "mean_toa_s": mean_toa_s,
                    }
                    for sf, share in zip(sf_values, sf_shares, strict=False):
                        row[f"sf{sf}_share_mean"] = share
                    rows.append(row)
    return rows


def _sample_step2_rows() -> list[dict[str, object]]:
    densities = [50, 100, 150]
    algos = ["ADR", "MixRA-H", "MixRA-Opt", "UCB1-SF"]
    clusters = list(DEFAULT_CONFIG.qos.clusters) + ["all"]
    rows: list[dict[str, object]] = []
    for snir_mode in SNIR_MODES:
        for algo_idx, algo in enumerate(algos):
            for density in densities:
                for cluster in clusters:
                    reward = max(0.2, 0.7 - 0.05 * algo_idx - 0.1 * (density - 0.5))
                    penalty = 0.05 if snir_mode == "snir_off" else 0.0
                    cluster_bonus = 0.03 if cluster == "gold" else 0.0
                    rows.append(
                        {
                            "density": density,  # Alias legacy de network_size.
                            "network_size": density,
                            "algo": algo,
                            "snir_mode": snir_mode,
                            "cluster": cluster,
                            "success_rate_mean": max(
                                0.3, 0.9 - 0.05 * algo_idx - penalty + cluster_bonus
                            ),
                            "bitrate_norm_mean": 0.4 + 0.1 * algo_idx - penalty,
                            "energy_norm_mean": 0.3 + 0.1 * algo_idx + penalty,
                            "reward_mean": reward - penalty + cluster_bonus,
                        }
                    )
    return rows


def _sample_selection_probs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_id in range(1, 11):
        for sf in (7, 8, 9, 10, 11, 12):
            rows.append(
                {
                    "round": round_id,
                    "sf": sf,
                    "selection_prob": max(0.05, 0.25 - 0.01 * (sf - 7) + 0.01 * round_id),
                }
            )
    return rows
