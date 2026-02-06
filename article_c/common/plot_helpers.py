"""Utilitaires de traçage pour les figures de l'article C."""

from __future__ import annotations

import csv
import logging
import math
import re
import warnings
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from article_c.common.plotting_style import (
    FIGURE_MARGINS,
    FIGURE_SIZE,
    LEGEND_MAX_HEIGHT_RATIO,
    LEGEND_MAX_WIDTH_RATIO,
    LEGEND_MIN_FONTSIZE,
    LEGEND_STYLE,
    SAVEFIG_STYLE,
    SUPTITLE_Y,
    adjust_legend_to_fit,
    apply_base_rcparams,
    apply_output_fonttype,
    legend_bbox_to_anchor,
    set_network_size_ticks,
    tight_layout_rect_from_margins,
)
from article_c.common.utils import ensure_dir
from article_c.common.config import DEFAULT_CONFIG

ALGO_LABELS = {
    "adr": "ADR",
    "apra": "APRA",
    "aimi": "Aimi",
    "loba": "LoBa",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
    "ucb1_sf": "UCB1-SF",
}
ALGO_COLORS = {
    "adr": "#1f77b4",
    "apra": "#8c564b",
    "aimi": "#17becf",
    "loba": "#9467bd",
    "mixra_h": "#ff7f0e",
    "mixra_opt": "#2ca02c",
    "ucb1_sf": "#d62728",
}
ALGO_MARKERS = {
    "adr": "o",
    "apra": "P",
    "aimi": "X",
    "loba": "v",
    "mixra_h": "s",
    "mixra_opt": "^",
    "ucb1_sf": "D",
}
ALGO_ALIASES = {
    "apra_like": "apra",
    "aimi_like": "aimi",
    "loba": "loba",
    "lo_ba": "loba",
    "lora_baseline": "loba",
    "lorawan_baseline": "loba",
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
BASE_FONT_FAMILY = "sans-serif"
BASE_FONT_SANS = ["DejaVu Sans", "Arial", "Liberation Sans"]
BASE_FONT_SIZE = 10
BASE_LINE_WIDTH = 1.6
BASE_GRID_COLOR = "#e0e0e0"
BASE_GRID_ALPHA = 0.6
BASE_GRID_LINEWIDTH = 0.8
BASE_DPI = 300
BASE_GRID_ENABLED = True
MAX_TIGHT_BBOX_SCALE = 4.0
MAX_TIGHT_BBOX_INCHES = 30.0
MAX_IMAGE_DIM_PX = 12000
MAX_IMAGE_TOTAL_PIXELS = 120_000_000
MAX_IEEE_FIGURE_SIZE_IN = (8.0, 6.0)
AXES_TITLE_Y = 1.02
SUPTITLE_TOP_RATIO = 0.85
FIGURE_SUBPLOT_TOP = FIGURE_MARGINS["top"]
FIGURE_SUBPLOT_BOTTOM = FIGURE_MARGINS["bottom"]
FIGURE_SUBPLOT_RIGHT = FIGURE_MARGINS.get("right", 0.98)
LEGEND_TOP_MARGIN = 0.74
LEGEND_TOP_RESERVED = 0.02
LEGEND_ROW_EXTRA_MARGIN = 0.05
LEGEND_ABOVE_TIGHT_LAYOUT_TOP = 0.86
LEGEND_RIGHT_MARGIN = 0.68
DEFAULT_LEGEND_LOC = "right"
CONSTANT_METRIC_VARIANCE_THRESHOLD = 1e-6
CONSTANT_METRIC_MESSAGE = "métrique constante – à investiguer"
MISSING_METRIC_MESSAGE = "données manquantes"
DEFAULT_EXPORT_FORMATS = ("png", "pdf")
_EXPORT_FORMATS = DEFAULT_EXPORT_FORMATS


class MetricStatus(str, Enum):
    OK = "ok"
    CONSTANT = "constant"
    MISSING = "missing"


@dataclass(frozen=True)
class MetricValues:
    values: list[float]
    status: MetricStatus


def apply_plot_style() -> None:
    """Applique un style homogène pour les figures Step1/Step2."""
    apply_base_rcparams()
    plt.rcParams.update(
        {
            "figure.figsize": FIGURE_SIZE,
            "figure.subplot.top": FIGURE_SUBPLOT_TOP,
            "figure.subplot.bottom": FIGURE_SUBPLOT_BOTTOM,
            "figure.subplot.right": FIGURE_SUBPLOT_RIGHT,
            "figure.dpi": BASE_DPI,
            "axes.grid": BASE_GRID_ENABLED,
            "axes.titley": AXES_TITLE_Y,
            "savefig.dpi": BASE_DPI,
        }
    )


def _normalize_export_formats(formats: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for fmt in formats:
        cleaned = str(fmt).strip().lstrip(".").lower()
        if cleaned:
            normalized.append(cleaned)
    if not normalized:
        raise ValueError("La liste des formats d'export est vide.")
    return tuple(dict.fromkeys(normalized))


def parse_export_formats(value: str | None) -> tuple[str, ...]:
    """Parse la valeur CLI --formats en liste normalisée."""
    if value is None:
        return DEFAULT_EXPORT_FORMATS
    parts = [part.strip() for part in value.split(",")]
    return _normalize_export_formats(parts)


def set_default_export_formats(formats: Iterable[str]) -> tuple[str, ...]:
    """Définit la liste globale des formats d'export."""
    global _EXPORT_FORMATS
    _EXPORT_FORMATS = _normalize_export_formats(formats)
    return _EXPORT_FORMATS
    plt.subplots_adjust(top=FIGURE_SUBPLOT_TOP)


def get_export_formats() -> tuple[str, ...]:
    """Retourne la liste globale des formats d'export."""
    return _EXPORT_FORMATS


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


def flatten_axes(axes: object) -> list[plt.Axes]:
    """Retourne une liste aplatie d'axes Matplotlib."""
    return _flatten_axes(axes)


def clear_axis_legends(axes: object) -> None:
    """Supprime proprement les légendes attachées aux axes."""
    for ax in _flatten_axes(axes):
        legend = ax.get_legend()
        if legend is None:
            continue
        legend.remove()
        ax.legend_ = None


def find_internal_legends(fig: plt.Figure) -> list[Legend]:
    """Retourne les légendes attachées aux axes d'une figure."""
    return [legend for ax in fig.axes if (legend := ax.get_legend()) is not None]


def collect_legend_entries(axes: object) -> tuple[list[Line2D], list[str]]:
    """Collecte toutes les entrées de légende depuis un ensemble d'axes."""
    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in _flatten_axes(axes):
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    return handles, labels


def deduplicate_legend_entries(
    handles: list[Line2D],
    labels: list[str],
) -> tuple[list[Line2D], list[str]]:
    """Déduplique les entrées de légende en conservant le premier handle."""
    seen: set[str] = set()
    dedup_handles: list[Line2D] = []
    dedup_labels: list[str] = []
    for handle, label in zip(handles, labels, strict=False):
        normalized_label = normalize_legend_label(label)
        if not normalized_label or normalized_label == "_nolegend_":
            continue
        if normalized_label in seen:
            continue
        seen.add(normalized_label)
        dedup_handles.append(handle)
        dedup_labels.append(normalized_label)
    return dedup_handles, dedup_labels


def normalize_legend_label(label: object) -> str:
    """Normalise les labels pour éviter les doublons incohérents."""
    cleaned = str(label).strip()
    if not cleaned:
        return ""
    normalized = cleaned
    for mode in ("on", "off"):
        normalized = re.sub(
            rf"\bsnir\s*[-_ ]?\s*{mode}\b",
            f"SNIR {mode}",
            normalized,
            flags=re.IGNORECASE,
        )
    normalized = re.sub(r"\bsnir\b", "SNIR", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


def _legend_bbox_ratios(legend: Legend) -> tuple[float, float] | None:
    fig = legend.figure
    if fig is None or fig.canvas is None:
        return None
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
    except Exception:
        return None
    fig_width, fig_height = fig.get_size_inches()
    if fig_width <= 0 or fig_height <= 0:
        return None
    width_ratio = bbox.width / (fig.dpi * fig_width)
    height_ratio = bbox.height / (fig.dpi * fig_height)
    return width_ratio, height_ratio


def _legend_needs_reflow(
    legend: Legend,
    *,
    max_width_ratio: float = LEGEND_MAX_WIDTH_RATIO,
    max_height_ratio: float = LEGEND_MAX_HEIGHT_RATIO,
) -> bool:
    ratios = _legend_bbox_ratios(legend)
    if ratios is None:
        return False
    width_ratio, height_ratio = ratios
    return width_ratio > max_width_ratio or height_ratio > max_height_ratio


def _legend_fallback_in_axis(
    *,
    axes: Iterable[plt.Axes] | None,
    handles: list[Line2D],
    labels: list[str],
    fontsize: float | None,
) -> Legend | None:
    axes_list = list(axes or [])
    if not axes_list:
        return None
    ax = axes_list[0]
    legend_kwargs: dict[str, object] = {
        "loc": "upper right",
        "frameon": True,
        "ncol": 1,
    }
    if fontsize is not None:
        legend_kwargs["fontsize"] = fontsize
    return ax.legend(handles, labels, **legend_kwargs)


def legend_ncols(legend: Legend, default: int) -> int:
    """Retourne le nombre de colonnes de légende avec un fallback robuste."""
    if hasattr(legend, "get_ncols"):
        try:
            return int(legend.get_ncols())
        except (TypeError, ValueError):
            pass
    if hasattr(legend, "_ncols"):
        try:
            return int(getattr(legend, "_ncols"))
        except (TypeError, ValueError):
            pass
    return default


def postprocess_legend(
    legend: Legend,
    *,
    legend_loc: str,
    handles: list[Line2D],
    labels: list[str],
    fig: plt.Figure,
    legend_ncols_default: int,
    axes: Iterable[plt.Axes] | None = None,
) -> tuple[Legend, int, bool]:
    """Ajuste la légende et bascule dans l'axe si nécessaire."""
    legend_cols = legend_ncols(legend, legend_ncols_default)
    legend_rows = max(1, math.ceil(len(labels) / max(1, legend_cols)))
    if adjust_legend_to_fit(legend):
        legend_cols = legend_ncols(legend, legend_ncols_default)
        legend_rows = max(1, math.ceil(len(labels) / max(1, legend_cols)))
        if _normalize_legend_loc(legend_loc) == "above":
            bbox_to_anchor = legend_bbox_to_anchor(
                legend=legend,
                legend_rows=legend_rows,
            )
            legend.set_bbox_to_anchor(bbox_to_anchor)
    if not _legend_needs_reflow(legend):
        return legend, legend_rows, False
    font_size: float | None = None
    texts = legend.get_texts()
    if texts:
        font_size = max(LEGEND_MIN_FONTSIZE, float(texts[0].get_fontsize()))
    legend.remove()
    fallback_legend = _legend_fallback_in_axis(
        axes=axes or fig.axes,
        handles=handles,
        labels=labels,
        fontsize=font_size,
    )
    if fallback_legend is None:
        return legend, legend_rows, False
    return fallback_legend, 1, True


def render_constant_metric(
    fig: plt.Figure,
    axes: object,
    *,
    message: str = CONSTANT_METRIC_MESSAGE,
    message_y: float | None = None,
    legend_loc: str = "right",
    show_fallback_legend: bool = True,
    legend_mode: str = "constante",
    legend_handles: tuple[list[Line2D], list[str]] | None = None,
) -> None:
    """Affiche un message centré lorsque la métrique est constante.

    legend_handles permet de fournir des handles/labels factices pour la légende.
    legend_mode="constante" conserve une légende existante ou réutilise les handles.
    """
    normalized_legend_mode = _normalize_legend_mode(legend_mode)
    handles: list[Line2D] = []
    labels: list[str] = []
    if normalized_legend_mode == "constante":
        if legend_handles is not None:
            handles, labels = legend_handles
        else:
            handles, labels = _collect_legend_handles(axes)
        if not handles and show_fallback_legend:
            handles, labels = fallback_legend_handles()
    for ax in _flatten_axes(axes):
        ax.clear()
        ax.axis("off")
    legend_style: dict[str, object] | None = None
    legend_rows = 1
    margins_for_layout: dict[str, float] | None = None
    if show_fallback_legend:
        should_add_legend = not _figure_has_legend(fig)
        if normalized_legend_mode == "constante":
            should_add_legend = should_add_legend or bool(handles)
        if should_add_legend:
            if not handles:
                if legend_handles is None:
                    handles, labels = fallback_legend_handles()
                else:
                    handles, labels = legend_handles
            if handles:
                handles, labels = deduplicate_legend_entries(handles, labels)
            if handles:
                legend_style, legend_rows = _legend_style(
                    legend_loc,
                    len(labels),
                    fig=fig,
                )
                margins_for_layout = legend_margins(
                    legend_loc,
                    legend_rows=legend_rows,
                    fig=fig,
                )
        elif normalized_legend_mode == "constante" and _figure_has_legend(fig):
            margins_for_layout = legend_margins(legend_loc, fig=fig)
    layout_rect = _layout_rect_from_margins(
        margins_for_layout,
        legend_rows=legend_rows,
        fig=fig,
    )
    message_x = (layout_rect[0] + layout_rect[2]) / 2
    message_y = message_y if message_y is not None else (layout_rect[1] + layout_rect[3]) / 2
    fig.text(
        message_x,
        message_y,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color="#444444",
    )
    if show_fallback_legend:
        should_add_legend = not _figure_has_legend(fig)
        if normalized_legend_mode == "constante":
            should_add_legend = should_add_legend or bool(handles)
        if should_add_legend:
            if handles:
                if legend_style is None:
                    legend_style, legend_rows = _legend_style(
                        legend_loc,
                        len(labels),
                        fig=fig,
                    )
                handles, labels = deduplicate_legend_entries(handles, labels)
                if handles:
                    legend = fig.legend(handles, labels, **legend_style)
                    legend, legend_rows, moved_to_axis = postprocess_legend(
                        legend,
                        legend_loc=legend_loc,
                        handles=handles,
                        labels=labels,
                        fig=fig,
                        legend_ncols_default=int(legend_style.get("ncol", 1)),
                        axes=_flatten_axes(axes),
                    )
                    if margins_for_layout is None:
                        margins_for_layout = (
                            FIGURE_MARGINS
                            if moved_to_axis
                            else legend_margins(
                                legend_loc,
                                legend_rows=legend_rows,
                                fig=fig,
                            )
                        )
                    apply_figure_layout(
                        fig,
                        margins=margins_for_layout,
                        bbox_to_anchor=None
                        if moved_to_axis
                        else legend.get_bbox_to_anchor(),
                        legend_rows=legend_rows,
                        legend_loc=legend_loc,
                    )
        elif normalized_legend_mode == "constante" and _figure_has_legend(fig):
            apply_figure_layout(
                fig,
                margins=legend_margins(legend_loc, fig=fig),
                legend_loc=legend_loc,
            )


def _metric_variance(
    values: list[float],
) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)


def is_constant_metric(
    values: MetricValues | list[float],
    *,
    threshold: float = CONSTANT_METRIC_VARIANCE_THRESHOLD,
) -> MetricStatus:
    if isinstance(values, MetricValues):
        if values.status is MetricStatus.MISSING:
            return MetricStatus.MISSING
        metric_values = values.values
    else:
        metric_values = values
    if not metric_values:
        return MetricStatus.CONSTANT
    if _metric_variance(metric_values) < threshold:
        return MetricStatus.CONSTANT
    return MetricStatus.OK


def metric_values(
    rows: list[dict[str, object]],
    metric_key: str,
) -> MetricValues:
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    values: list[float] = []
    column_present = any(median_key in row for row in rows)
    for row in rows:
        value = row.get(median_key)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    status = MetricStatus.OK if column_present else MetricStatus.MISSING
    return MetricValues(values=values, status=status)


def metric_status_message(status: MetricStatus) -> str:
    if status is MetricStatus.MISSING:
        return MISSING_METRIC_MESSAGE
    if status is MetricStatus.CONSTANT:
        return CONSTANT_METRIC_MESSAGE
    return ""


def render_metric_status(
    fig: plt.Figure,
    axes: object,
    status: MetricStatus,
    *,
    message_y: float | None = None,
    legend_loc: str = "right",
    show_fallback_legend: bool = True,
    legend_mode: str = "constante",
    legend_handles: tuple[list[Line2D], list[str]] | None = None,
) -> None:
    if status is MetricStatus.OK:
        return
    if (
        show_fallback_legend
        and legend_handles is None
        and not _collect_legend_handles(axes)[0]
    ):
        legend_handles = fallback_legend_handles()
    render_constant_metric(
        fig,
        axes,
        message=metric_status_message(status),
        message_y=message_y,
        legend_loc=legend_loc,
        show_fallback_legend=show_fallback_legend,
        legend_mode=legend_mode,
        legend_handles=legend_handles,
    )


def select_received_metric_key(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    derived_key: str = RECEIVED_ALGO_MEAN_KEY,
    tolerance: float = RECEIVED_ALGO_TOL,
) -> str:
    """Retourne la clé de métrique à utiliser pour les métriques de réception.

    Source officielle: la valeur "received" est dérivée de sent_mean * pdr_mean.
    La moyenne agrégée "received_mean" peut s'écarter de cette valeur (moyenne
    d'un produit ≠ produit des moyennes). On calcule donc received_algo_mean à
    partir de ces champs et on documente l'écart si besoin avant d'utiliser
    systématiquement la valeur dérivée.
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
            "received_mean ne correspond pas partout à sent_mean*pdr_mean "
            f"(écart max={max(differences):.6g}); "
            "utilisation de received_algo_mean dérivé.",
            stacklevel=2,
        )
    return derived_key


def place_legend(ax: plt.Axes, *, legend_loc: str | None = None) -> None:
    """Place la légende selon les consignes (au-dessus ou à droite)."""
    legend_loc = legend_loc or DEFAULT_LEGEND_LOC
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        handles, labels = fallback_legend_handles()
    if handles:
        handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        return
    legend_style, legend_rows = _legend_style(
        legend_loc,
        len(labels),
        fig=ax.figure,
    )
    legend = ax.figure.legend(handles, labels, **legend_style)
    legend, legend_rows, moved_to_axis = postprocess_legend(
        legend,
        legend_loc=legend_loc,
        handles=handles,
        labels=labels,
        fig=ax.figure,
        legend_ncols_default=int(legend_style.get("ncol", 1)),
        axes=[ax],
    )
    apply_figure_layout(
        ax.figure,
        margins=FIGURE_MARGINS
        if moved_to_axis
        else legend_margins(legend_loc, legend_rows=legend_rows, fig=ax.figure),
        bbox_to_anchor=None if moved_to_axis else legend.get_bbox_to_anchor(),
        legend_rows=legend_rows,
        legend_loc=legend_loc,
    )


def _legend_style(
    legend_loc: str,
    label_count: int | None = None,
    fig: plt.Figure | None = None,
) -> tuple[dict[str, object], int]:
    normalized = _normalize_legend_loc(legend_loc)
    if normalized == "right":
        return {
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "ncol": 1,
            "frameon": False,
        }, 1
    legend_style = dict(LEGEND_STYLE)
    legend_rows = 1
    if label_count is not None:
        ncol = int(legend_style.get("ncol", label_count) or 1)
        ncol = min(label_count, max(1, ncol))
        ncol, legend_rows = _legend_layout_from_fig(fig, label_count, ncol)
        legend_style["ncol"] = ncol
    legend_style["bbox_to_anchor"] = legend_bbox_to_anchor(legend_rows=legend_rows)
    return legend_style, legend_rows


def _figure_size(fig: plt.Figure | None) -> tuple[float, float]:
    if fig is None:
        return FIGURE_SIZE
    try:
        width, height = fig.get_size_inches()
    except (AttributeError, TypeError, ValueError):
        return FIGURE_SIZE
    if not width or not height:
        return FIGURE_SIZE
    return float(width), float(height)


def _scale_margin_from_base(value: float, fig: plt.Figure | None) -> float:
    _, fig_height = _figure_size(fig)
    base_height = FIGURE_SIZE[1]
    if fig_height <= 0 or base_height <= 0:
        return value
    return max(0.0, min(1.0, value * base_height / fig_height))


def _legend_layout_from_fig(
    fig: plt.Figure | None,
    label_count: int,
    ncol: int,
) -> tuple[int, int]:
    if label_count <= 0:
        return max(1, ncol), 1
    fig_width, _ = _figure_size(fig)
    base_width = FIGURE_SIZE[0]
    if base_width <= 0:
        width_ratio = 1.0
    else:
        width_ratio = fig_width / base_width
    width_ratio = max(0.5, width_ratio)
    effective_ncol = max(1, min(label_count, int(math.floor(ncol * width_ratio))))
    legend_rows = max(1, math.ceil(label_count / effective_ncol))
    return effective_ncol, legend_rows


def _legend_top_margin(fig: plt.Figure | None, legend_rows: int) -> float:
    extra_rows = max(0, legend_rows - 1)
    base_margin = max(0.0, LEGEND_TOP_MARGIN - LEGEND_ROW_EXTRA_MARGIN * extra_rows)
    return _scale_margin_from_base(base_margin, fig)


def _legend_top_reserved(fig: plt.Figure | None, legend_rows: int) -> float:
    base_reserved = LEGEND_TOP_RESERVED * max(1, legend_rows)
    return _scale_margin_from_base(base_reserved, fig)


def _legend_right_margin(fig: plt.Figure | None) -> float:
    fig_width, _ = _figure_size(fig)
    base_width = FIGURE_SIZE[0]
    if fig_width <= 0 or base_width <= 0:
        return LEGEND_RIGHT_MARGIN
    margin_inches = (1.0 - LEGEND_RIGHT_MARGIN) * base_width
    right = 1.0 - margin_inches / fig_width
    return max(0.0, min(1.0, right))


def _legend_margins(
    legend_loc: str,
    *,
    legend_rows: int = 1,
    fig: plt.Figure | None = None,
) -> dict[str, float]:
    normalized = _normalize_legend_loc(legend_loc)
    if normalized == "right":
        return {
            "top": FIGURE_SUBPLOT_TOP,
            "right": min(FIGURE_SUBPLOT_RIGHT, _legend_right_margin(fig)),
        }
    if normalized == "above":
        return {"top": _legend_top_margin(fig, legend_rows)}
    return {"top": FIGURE_SUBPLOT_TOP}


def legend_margins(
    legend_loc: str,
    *,
    legend_rows: int = 1,
    fig: plt.Figure | None = None,
) -> dict[str, float]:
    """Expose les marges recommandées pour une légende donnée."""
    return _legend_margins(legend_loc, legend_rows=legend_rows, fig=fig)


def suptitle_y_from_top(
    fig: plt.Figure,
    *,
    fallback: float = SUPTITLE_Y,
    ratio: float = SUPTITLE_TOP_RATIO,
) -> float:
    """Calcule la position verticale du suptitle selon la marge supérieure réelle."""
    try:
        top = float(fig.subplotpars.top)
    except (AttributeError, TypeError, ValueError):
        return fallback
    top = max(0.0, min(1.0, top))
    if top >= 1.0:
        return fallback
    return min(0.99, top + (1.0 - top) * ratio)


def _normalize_legend_loc(legend_loc: str) -> str:
    normalized = str(legend_loc or "").strip().lower()
    if normalized in {"haut", "top", "above"}:
        return "above"
    if normalized in {"droite", "right"}:
        return "right"
    return normalized


def _normalize_legend_mode(legend_mode: str) -> str:
    normalized = str(legend_mode or "").strip().lower()
    if normalized in {"constante", "constant", "persist"}:
        return "constante"
    return normalized or "fallback"


def _figure_has_legend(fig: plt.Figure) -> bool:
    if fig.legends:
        return True
    return any(ax.get_legend() is not None for ax in fig.axes)


def assert_legend_present(fig: plt.Figure, context: str) -> None:
    """Émet un warning si aucune légende n'est détectée."""
    if _figure_has_legend(fig):
        return
    context_label = str(context or "").strip() or "figure inconnue"
    LOGGER.warning("Aucune légende détectée pour %s.", context_label)


def _collect_legend_handles(axes: object) -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in _flatten_axes(axes):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    return handles, labels


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
    legend_loc: str | None = None,
    handles: list[Line2D] | None = None,
    labels: list[str] | None = None,
    use_fallback: bool = True,
) -> None:
    """Ajoute une légende globale à la figure."""
    legend_loc = legend_loc or DEFAULT_LEGEND_LOC
    if _normalize_legend_loc(legend_loc) == "right":
        clear_axis_legends(fig.axes)
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles and use_fallback:
        handles, labels = fallback_legend_handles()
    if handles:
        handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        return
    legend_style, legend_rows = _legend_style(
        legend_loc,
        len(labels),
        fig=fig,
    )
    legend = fig.legend(handles, labels, **legend_style)
    bbox_to_anchor = legend_bbox_to_anchor(legend=legend, legend_rows=legend_rows)
    legend.set_bbox_to_anchor(bbox_to_anchor)
    legend, legend_rows, moved_to_axis = postprocess_legend(
        legend,
        legend_loc=legend_loc,
        handles=handles,
        labels=labels,
        fig=fig,
        legend_ncols_default=int(legend_style.get("ncol", 1)),
        axes=[ax],
    )
    apply_figure_layout(
        fig,
        margins=FIGURE_MARGINS
        if moved_to_axis
        else legend_margins(legend_loc, legend_rows=legend_rows, fig=fig),
        bbox_to_anchor=None if moved_to_axis else legend.get_bbox_to_anchor(),
        legend_rows=legend_rows,
        legend_loc=legend_loc,
    )


def add_figure_legend(
    fig: plt.Figure,
    handles: list[Line2D],
    labels: list[str],
    *,
    legend_loc: str | None = None,
) -> int:
    """Ajoute une légende globale à la figure et applique les marges associées."""
    legend_loc = legend_loc or DEFAULT_LEGEND_LOC
    if not handles:
        return 0
    handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        return 0
    legend_style, legend_rows = _legend_style(
        legend_loc,
        len(labels),
        fig=fig,
    )
    legend = fig.legend(handles, labels, **legend_style)
    bbox_to_anchor = legend_bbox_to_anchor(legend=legend, legend_rows=legend_rows)
    legend.set_bbox_to_anchor(bbox_to_anchor)
    legend, legend_rows, moved_to_axis = postprocess_legend(
        legend,
        legend_loc=legend_loc,
        handles=handles,
        labels=labels,
        fig=fig,
        legend_ncols_default=int(legend_style.get("ncol", 1)),
    )
    apply_figure_layout(
        fig,
        margins=FIGURE_MARGINS
        if moved_to_axis
        else legend_margins(legend_loc, legend_rows=legend_rows, fig=fig),
        bbox_to_anchor=None if moved_to_axis else legend.get_bbox_to_anchor(),
        legend_rows=legend_rows,
        legend_loc=legend_loc,
    )
    return legend_rows


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    use_tight: bool = False,
    formats: Iterable[str] | None = None,
    bbox_inches: str | bool | None = None,
) -> None:
    """Sauvegarde la figure dans les formats demandés."""
    ensure_dir(output_dir)
    apply_output_fonttype()
    if use_tight:
        apply_figure_layout(fig, tight_layout=True)
    selected_formats = _EXPORT_FORMATS if formats is None else _normalize_export_formats(formats)
    effective_bbox = bbox_inches
    if effective_bbox is None:
        effective_bbox = "tight"
    for ext in selected_formats:
        save_figure_path(
            fig,
            output_dir / f"{stem}.{ext}",
            fmt=ext,
            bbox_inches=effective_bbox,
        )


def _alpha_is_transparent(alpha: object) -> bool:
    if alpha is None:
        return False
    if isinstance(alpha, (list, tuple)):
        return any(item is not None and item < 1 for item in alpha)
    if hasattr(alpha, "__iter__") and not isinstance(alpha, (str, bytes)):
        try:
            return any(item is not None and float(item) < 1 for item in alpha)
        except TypeError:
            return False
    try:
        return float(alpha) < 1
    except (TypeError, ValueError):
        return False


@contextmanager
def _rasterize_transparent_artists(fig: plt.Figure) -> Iterable[None]:
    modified: list[tuple[object, bool]] = []
    for artist in fig.findobj():
        if not hasattr(artist, "get_alpha") or not hasattr(artist, "set_rasterized"):
            continue
        if _alpha_is_transparent(artist.get_alpha()):
            try:
                previous = bool(artist.get_rasterized())
            except Exception:
                previous = False
            try:
                artist.set_rasterized(True)
                modified.append((artist, previous))
            except Exception:
                continue
    try:
        yield
    finally:
        for artist, previous in modified:
            try:
                artist.set_rasterized(previous)
            except Exception:
                continue


@contextmanager
def _force_opaque_alpha(fig: plt.Figure) -> Iterable[None]:
    modified: list[tuple[object, object]] = []
    for artist in fig.findobj():
        if not hasattr(artist, "get_alpha") or not hasattr(artist, "set_alpha"):
            continue
        alpha = artist.get_alpha()
        if not _alpha_is_transparent(alpha):
            continue
        try:
            artist.set_alpha(1)
            modified.append((artist, alpha))
        except Exception:
            continue
    try:
        yield
    finally:
        for artist, previous in modified:
            try:
                artist.set_alpha(previous)
            except Exception:
                continue


def save_figure_path(
    fig: plt.Figure,
    output_path: Path,
    *,
    fmt: str | None = None,
    bbox_inches: str | bool | None = None,
) -> None:
    """Sauvegarde une figure en gérant l'export EPS (transparences rasterisées)."""
    _apply_figure_size_clamp(fig)
    ensure_dir(output_path.parent)
    format_name = fmt or output_path.suffix.lstrip(".").lower()
    savefig_style = dict(SAVEFIG_STYLE)
    avoid_tight_bbox = bool(getattr(fig, "_avoid_tight_bbox", False))
    if bbox_inches is None:
        bbox_inches = "tight" if not avoid_tight_bbox else False
    if avoid_tight_bbox and bbox_inches == "tight":
        bbox_inches = False
    if bbox_inches is not False:
        safe_bbox = _safe_bbox_inches(fig, bbox_inches)
        savefig_style["bbox_inches"] = safe_bbox
    safe_dpi = _safe_dpi(fig, BASE_DPI)
    if format_name == "eps":
        with _rasterize_transparent_artists(fig):
            with _force_opaque_alpha(fig):
                fig.savefig(output_path, format=format_name, dpi=safe_dpi, **savefig_style)
    else:
        fig.savefig(output_path, format=format_name, dpi=safe_dpi, **savefig_style)


def _safe_bbox_inches(
    fig: plt.Figure,
    bbox_inches: str | bool | None,
) -> str | None:
    if bbox_inches is False:
        return None
    if bbox_inches != "tight":
        return bbox_inches if isinstance(bbox_inches, str) or bbox_inches is None else None
    if fig.canvas is None:
        return bbox_inches
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer=renderer)
    except Exception as exc:
        LOGGER.debug("Impossible d'évaluer la bounding box tight: %s", exc)
        return bbox_inches
    if tight_bbox is None:
        return bbox_inches
    fig_width_in, fig_height_in = fig.get_size_inches()
    bbox_width_in = tight_bbox.width / fig.dpi
    bbox_height_in = tight_bbox.height / fig.dpi
    max_width_in = max(fig_width_in * MAX_TIGHT_BBOX_SCALE, MAX_TIGHT_BBOX_INCHES)
    max_height_in = max(fig_height_in * MAX_TIGHT_BBOX_SCALE, MAX_TIGHT_BBOX_INCHES)
    if bbox_width_in > max_width_in or bbox_height_in > max_height_in:
        LOGGER.warning(
            "Bounding box tight trop grande (%.2f x %.2f in, limite %.2f x %.2f). "
            "Désactivation de bbox_inches.",
            bbox_width_in,
            bbox_height_in,
            max_width_in,
            max_height_in,
        )
        return None
    return bbox_inches


def _safe_dpi(fig: plt.Figure, dpi: float) -> float:
    fig_width_in, fig_height_in = fig.get_size_inches()
    width_px = fig_width_in * dpi
    height_px = fig_height_in * dpi
    max_dim = max(width_px, height_px)
    total_pixels = width_px * height_px
    max_dpi = dpi
    if max_dim > MAX_IMAGE_DIM_PX:
        max_dpi = min(max_dpi, MAX_IMAGE_DIM_PX / max(fig_width_in, fig_height_in))
    if total_pixels > MAX_IMAGE_TOTAL_PIXELS:
        max_dpi = min(max_dpi, math.sqrt(MAX_IMAGE_TOTAL_PIXELS / (fig_width_in * fig_height_in)))
    if max_dpi < dpi:
        LOGGER.warning(
            "DPI réduit de %.1f à %.1f pour éviter une image trop grande (%.0f x %.0f px).",
            dpi,
            max_dpi,
            width_px,
            height_px,
        )
    return max_dpi


def _apply_figure_size_clamp(
    fig: plt.Figure,
    *,
    max_size: tuple[float, float] = MAX_IEEE_FIGURE_SIZE_IN,
) -> bool:
    fig_width_in, fig_height_in = fig.get_size_inches()
    max_width, max_height = max_size
    clamped_width = min(fig_width_in, max_width)
    clamped_height = min(fig_height_in, max_height)
    if clamped_width == fig_width_in and clamped_height == fig_height_in:
        return False
    LOGGER.warning(
        "Taille de figure clampée à %.2f x %.2f in (était %.2f x %.2f in).",
        clamped_width,
        clamped_height,
        fig_width_in,
        fig_height_in,
    )
    fig.set_size_inches(clamped_width, clamped_height, forward=True)
    return True


def apply_figure_layout(
    fig: plt.Figure,
    *,
    figsize: tuple[float, float] | None = None,
    tight_layout: bool | Mapping[str, object] = False,
    bbox_to_anchor: tuple[float, float] | None = None,
    margins: dict[str, float] | None = None,
    legend_rows: int = 1,
    legend_loc: str | None = None,
) -> None:
    """Applique taille, marges, légendes et tight_layout sur une figure."""
    extra_legend_rows = max(0, legend_rows - 1)
    reserved_top = 0.0
    if figsize is not None:
        fig.set_size_inches(*figsize, forward=True)
    _apply_figure_size_clamp(fig)
    if margins is None:
        margins = dict(FIGURE_MARGINS)
    normalized_legend_loc = _normalize_legend_loc(legend_loc) if legend_loc else ""
    if normalized_legend_loc == "right":
        margins.update(_legend_margins("right", legend_rows=legend_rows, fig=fig))
    elif fig.legends:
        margins["top"] = max(
            margins.get("top", FIGURE_MARGINS["top"]),
            _legend_top_margin(fig, legend_rows),
        )
    if normalized_legend_loc == "right":
        right_margin = _legend_right_margin(fig)
        if "right" not in margins or margins["right"] > right_margin:
            margins["right"] = right_margin
    if margins:
        adjusted_margins = dict(margins)
        if "top" in adjusted_margins:
            reserved_top = min(
                _legend_top_reserved(fig, legend_rows),
                adjusted_margins["top"],
            )
        fig.subplots_adjust(**adjusted_margins)
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
                    max(
                        0.0,
                        top
                        - _scale_margin_from_base(
                            LEGEND_ROW_EXTRA_MARGIN * extra_legend_rows,
                            fig,
                        ),
                    ),
                )
            if reserved_top and adjusted_tight.get("rect"):
                left, bottom, right, top = adjusted_tight["rect"]
                adjusted_tight["rect"] = (
                    left,
                    bottom,
                    right,
                    max(0.0, top - reserved_top),
                )
            if normalized_legend_loc == "right" and adjusted_tight.get("rect"):
                left, bottom, right, top = adjusted_tight["rect"]
                adjusted_tight["rect"] = (
                    left,
                    bottom,
                    min(right, _legend_right_margin(fig)),
                    top,
                )
            fig.tight_layout(**adjusted_tight)
        else:
            rect = tight_layout_rect_from_margins(margins)
            fig.tight_layout(rect=rect)
    fig._avoid_tight_bbox = normalized_legend_loc == "right"


def _layout_rect_from_margins(
    margins: dict[str, float] | None,
    *,
    legend_rows: int = 1,
    fig: plt.Figure | None = None,
) -> tuple[float, float, float, float]:
    if not margins:
        return (0.0, 0.0, 1.0, 1.0)
    adjusted_margins = dict(margins)
    reserved_top = 0.0
    if "top" in adjusted_margins:
        reserved_top = min(
            _legend_top_reserved(fig, legend_rows),
            adjusted_margins["top"],
        )
    return (
        adjusted_margins.get("left", 0.0),
        adjusted_margins.get("bottom", 0.0),
        adjusted_margins.get("right", 1.0),
        max(0.0, adjusted_margins.get("top", 1.0) - reserved_top),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def safe_load_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        warnings.warn(f"CSV introuvable: {path}", stacklevel=2)
        return None
    try:
        rows = _read_csv_rows(path)
    except OSError as exc:
        warnings.warn(f"CSV illisible ({path}): {exc}", stacklevel=2)
        return None
    if not rows:
        warnings.warn(f"CSV vide: {path}", stacklevel=2)
        return None
    return rows


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
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALGO_ALIASES.get(normalized, normalized)


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
    rows = safe_load_csv(source_path)
    if not rows:
        return []
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
    base_rows = _count_csv_data_rows(path) if path.exists() else 0
    if by_round.exists() and _is_intermediate_complete(by_round, base_rows):
        return by_round
    by_replication = path.with_name("aggregated_results_by_replication.csv")
    if by_replication.exists() and _is_intermediate_complete(
        by_replication, base_rows
    ):
        return by_replication
    return None


def _count_csv_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            total_lines = sum(1 for _ in handle)
    except OSError:
        return 0
    return max(0, total_lines - 1)


def _is_intermediate_complete(path: Path, base_rows: int) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
    except OSError:
        return False
    rows = _count_csv_data_rows(path)
    if rows == 0:
        return False
    if base_rows and rows < base_rows:
        return False
    return True


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
    rows = safe_load_csv(path)
    if not rows:
        return []
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


def warn_if_insufficient_network_sizes(network_sizes: Iterable[int]) -> None:
    """Avertit si une seule taille de réseau est disponible."""
    sizes = list(network_sizes)
    if len(sizes) < 2:
        warnings.warn(
            "Moins de deux tailles de réseau disponibles ; tracé avec une seule valeur.",
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
