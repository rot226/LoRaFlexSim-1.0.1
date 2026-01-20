"""Style de tracé (placeholder)."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

PLOT_STYLE = {
    "figure.figsize": (7.2, 4.2),
    "figure.subplot.top": 0.80,
    "axes.grid": True,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
}

LEGEND_STYLE = {
    "loc": "lower center",
    "bbox_to_anchor": (0.5, 1.02),
    "ncol": 3,
    "frameon": False,
}

SAVEFIG_STYLE = {
    "bbox_inches": None,
}


def apply_plot_style() -> None:
    """Applique la taille de figure et la marge supérieure demandées."""
    plt.rcParams.update(PLOT_STYLE)
    plt.subplots_adjust(top=PLOT_STYLE["figure.subplot.top"])


def set_network_size_ticks(ax: plt.Axes, network_sizes: Iterable[int]) -> None:
    """Force les ticks de tailles de réseau et les formate en entier."""
    ax.set_xticks(list(network_sizes))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
