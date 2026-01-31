"""Style de tracé (placeholder)."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.legend import Legend

IEEE_STYLE = True

BASE_FIGURE_SIZE = (7.2, 4.2)
IEEE_FIGURE_SIZE = (7.16, 4.77)

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
}
IEEE_FIGURE_MARGINS = {
    "top": 0.84,
    "bottom": 0.14,
}

BASE_SUPTITLE_Y = 0.965
IEEE_SUPTITLE_Y = 0.945

FIGURE_SIZE = IEEE_FIGURE_SIZE if IEEE_STYLE else BASE_FIGURE_SIZE
LEGEND_STYLE = IEEE_LEGEND_STYLE if IEEE_STYLE else BASE_LEGEND_STYLE
FIGURE_MARGINS = IEEE_FIGURE_MARGINS if IEEE_STYLE else BASE_FIGURE_MARGINS
SUPTITLE_Y = IEEE_SUPTITLE_Y if IEEE_STYLE else BASE_SUPTITLE_Y

SAVEFIG_STYLE = {
    "bbox_inches": None,
}

LEGEND_ANCHOR_BASE_Y = 1.0
LEGEND_ANCHOR_PADDING = 0.01
LEGEND_ROW_HEIGHT = 0.045
OUTPUT_FONT_TYPES = {
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}


def apply_output_fonttype() -> None:
    """Force l'export en police TrueType pour PS/PDF."""
    plt.rcParams.update(OUTPUT_FONT_TYPES)


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


def legend_extra_height(fig_height_in: float, legend_rows: int) -> float:
    """Calcule un supplément de hauteur (en pouces) selon le nombre de lignes de légende."""
    if fig_height_in <= 0:
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


def set_network_size_ticks(ax: plt.Axes, network_sizes: Iterable[int]) -> None:
    """Force les ticks de tailles de réseau et les formate en entier."""
    ax.set_xticks(list(network_sizes))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
