"""Style de tracé (placeholder)."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

LEGEND_STYLE = {
    "loc": "lower center",
    "bbox_to_anchor": (0.5, 1.10),
    "ncol": 4,
    "frameon": False,
}

SAVEFIG_STYLE = {
    "bbox_inches": None,
}


def set_network_size_ticks(ax: plt.Axes, network_sizes: Iterable[int]) -> None:
    """Force les ticks de tailles de réseau et les formate en entier."""
    ax.set_xticks(list(network_sizes))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
