"""Style de tracé partagé pour les figures de l'article C."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.legend import Legend

IEEE_STYLE = True

SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.5
HEIGHT_RATIO = 0.8
SINGLE_COLUMN_FIGSIZE = (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * HEIGHT_RATIO)
DOUBLE_COLUMN_FIGSIZE = (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * HEIGHT_RATIO)
BASE_FIGURE_SIZE = DOUBLE_COLUMN_FIGSIZE
IEEE_FIGURE_SIZE = DOUBLE_COLUMN_FIGSIZE

BASE_LEGEND_STYLE = {
    "loc": "upper center",
    "ncol": 4,
    "frameon": False,
}
IEEE_LEGEND_STYLE = {
    "loc": "upper center",
    "ncol": 3,
    "frameon": False,
    "columnspacing": 1.1,
}

BASE_FIGURE_MARGINS = {
    "top": 0.88,
    "bottom": 0.12,
    "right": 0.98,
}
IEEE_FIGURE_MARGINS = {
    "top": 0.84,
    "bottom": 0.14,
    "right": 0.79,
}
BASE_TIGHT_LAYOUT_RECT = (0.08, 0.12, 0.98, 0.92)
IEEE_TIGHT_LAYOUT_RECT = (0.07, 0.12, 0.8, 0.9)

BASE_SUPTITLE_Y = 0.965
IEEE_SUPTITLE_Y = 0.945

FIGURE_SIZE = IEEE_FIGURE_SIZE if IEEE_STYLE else BASE_FIGURE_SIZE
LEGEND_STYLE = IEEE_LEGEND_STYLE if IEEE_STYLE else BASE_LEGEND_STYLE
FIGURE_MARGINS = IEEE_FIGURE_MARGINS if IEEE_STYLE else BASE_FIGURE_MARGINS
SUPTITLE_Y = IEEE_SUPTITLE_Y if IEEE_STYLE else BASE_SUPTITLE_Y
TIGHT_LAYOUT_RECT = IEEE_TIGHT_LAYOUT_RECT if IEEE_STYLE else BASE_TIGHT_LAYOUT_RECT

SAVEFIG_STYLE = {
    "pad_inches": 0.04,
}

LEGEND_ANCHOR_BASE_Y = 1.0
LEGEND_ANCHOR_PADDING = 0.01
LEGEND_ROW_HEIGHT = 0.045
LEGEND_MAX_WIDTH_RATIO = 0.98
LEGEND_MAX_HEIGHT_RATIO = 0.22
LEGEND_MIN_FONTSIZE = 8
OUTPUT_FONT_TYPES = {
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}
BASE_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 6.0,
    "lines.markeredgewidth": 0.8,
    "figure.figsize": FIGURE_SIZE,
    "figure.constrained_layout.use": False,
}


def apply_base_rcparams() -> None:
    """Applique les rcParams homogènes pour les figures."""
    plt.rcParams.update(BASE_RCPARAMS)


def apply_ieee_style(*, use_constrained_layout: bool = False) -> None:
    """Applique le style IEEE et sélectionne un seul système de layout."""
    apply_base_rcparams()
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": bool(use_constrained_layout),
        }
    )


def apply_output_fonttype() -> None:
    """Force l'export en police TrueType pour PS/PDF."""
    plt.rcParams.update(OUTPUT_FONT_TYPES)


def parse_export_formats(value: str | None) -> tuple[str, ...]:
    """Expose parse_export_formats via plot_helpers (compatibilité)."""
    from article_c.common.plot_helpers import (
        parse_export_formats as _parse_export_formats,
    )

    return _parse_export_formats(value)


def set_default_export_formats(formats: Iterable[str]) -> tuple[str, ...]:
    """Expose set_default_export_formats via plot_helpers (compatibilité)."""
    from article_c.common.plot_helpers import (
        set_default_export_formats as _set_default_export_formats,
    )

    return _set_default_export_formats(formats)


def _legend_height_in_figure(legend: Legend) -> float | None:
    fig = legend.figure
    if fig is None or fig.canvas is None:
        return None
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
    except Exception:
        return None
    fig_height_in = fig.get_size_inches()[1]
    if fig_height_in <= 0:
        return None
    legend_height_in = bbox.height / fig.dpi
    return legend_height_in / fig_height_in


def legend_extra_height(
    fig_height_in: float,
    legend_rows: int,
    *,
    legend_loc: str | None = None,
) -> float:
    """Calcule un supplément de hauteur (en pouces) selon le nombre de lignes de légende."""
    if fig_height_in <= 0:
        return 0.0
    normalized_loc = str(legend_loc or "above").strip().lower()
    if normalized_loc not in {"above", "haut", "top"}:
        return 0.0
    return fig_height_in * LEGEND_ROW_HEIGHT * max(1, legend_rows)


def legend_bbox_to_anchor(
    *,
    legend: Legend | None = None,
    legend_rows: int = 1,
    anchor_x: float = 0.5,
) -> tuple[float, float]:
    """Calcule un bbox_to_anchor dynamique pour une légende au-dessus."""
    legend_height = None
    if legend is not None:
        legend_height = _legend_height_in_figure(legend)
    if legend_height is None:
        legend_height = LEGEND_ROW_HEIGHT * max(1, legend_rows)
    return (anchor_x, LEGEND_ANCHOR_BASE_Y + LEGEND_ANCHOR_PADDING + legend_height)


def _legend_ncols(legend: Legend, default: int | None = None) -> int:
    if default is None:
        default = int(LEGEND_STYLE.get("ncol", 1) or 1)
    get_ncols = getattr(legend, "get_ncols", None)
    ncols = None
    if callable(get_ncols):
        try:
            ncols = int(get_ncols())
        except (TypeError, ValueError):
            ncols = None
    if ncols is None:
        ncols = getattr(legend, "_ncols", None)
    try:
        ncols = int(ncols) if ncols is not None else default
    except (TypeError, ValueError):
        ncols = default
    return max(1, ncols)


def set_network_size_ticks(ax: plt.Axes, network_sizes: Iterable[int]) -> None:
    """Force les ticks de tailles de réseau et les formate en entier."""
    ax.set_xticks(list(network_sizes))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))


def tight_layout_rect_from_margins(
    margins: dict[str, float] | None = None,
    *,
    fallback: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Construit un rect pour tight_layout à partir de marges normalisées."""
    base = TIGHT_LAYOUT_RECT if fallback is None else fallback
    if not margins:
        return base
    return (
        margins.get("left", base[0]),
        margins.get("bottom", base[1]),
        margins.get("right", base[2]),
        margins.get("top", base[3]),
    )


def adjust_legend_to_fit(
    legend: Legend,
    *,
    max_width_ratio: float = LEGEND_MAX_WIDTH_RATIO,
    max_height_ratio: float = LEGEND_MAX_HEIGHT_RATIO,
    min_fontsize: float = LEGEND_MIN_FONTSIZE,
    max_attempts: int = 3,
) -> bool:
    """Réduit ncol/fontsize si la légende dépasse la zone utile."""
    fig = legend.figure
    if fig is None or fig.canvas is None:
        return False
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return False
    adjusted = False
    for _ in range(max_attempts):
        bbox = legend.get_window_extent(renderer=renderer)
        fig_width, fig_height = fig.get_size_inches()
        if fig_width <= 0 or fig_height <= 0:
            break
        width_ratio = bbox.width / (fig.dpi * fig_width)
        height_ratio = bbox.height / (fig.dpi * fig_height)
        if width_ratio <= max_width_ratio and height_ratio <= max_height_ratio:
            break
        current_ncols = _legend_ncols(legend)
        if width_ratio > max_width_ratio and current_ncols > 1:
            legend.set_ncols(max(1, current_ncols - 1))
            adjusted = True
        else:
            texts = legend.get_texts()
            if not texts:
                break
            current_size = float(texts[0].get_fontsize())
            if current_size <= min_fontsize:
                break
            new_size = max(min_fontsize, current_size - 1)
            for text in texts:
                text.set_fontsize(new_size)
            adjusted = True
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            break
    return adjusted
