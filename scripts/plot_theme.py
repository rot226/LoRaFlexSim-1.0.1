"""Thème matplotlib partagé pour les scripts de visualisation."""

from __future__ import annotations

from typing import Any, Mapping

from article_c.common.plotting_style import apply_base_rcparams

SNIR_COLORS: Mapping[str, str] = {
    "snir_on": "#d62728",
    "snir_off": "#1f77b4",
    "snir_unknown": "#7f7f7f",
}

THEME_FONT_SIZE = 10
THEME_TITLE_SIZE = 12
THEME_LABEL_SIZE = 11
THEME_TICK_LABEL_SIZE = 10
THEME_LEGEND_SIZE = 10
THEME_LINE_WIDTH = 2.0
THEME_MARKER_SIZE = 6.0
THEME_MARKER_EDGE_WIDTH = 0.8


def apply_plot_theme(plt: Any) -> None:
    """Applique un thème matplotlib partagé (polices, lignes, marqueurs)."""
    apply_base_rcparams()
    plt.rcParams.update(
        {
            "font.size": THEME_FONT_SIZE,
            "axes.titlesize": THEME_TITLE_SIZE,
            "axes.labelsize": THEME_LABEL_SIZE,
            "legend.fontsize": THEME_LEGEND_SIZE,
            "xtick.labelsize": THEME_TICK_LABEL_SIZE,
            "ytick.labelsize": THEME_TICK_LABEL_SIZE,
            "lines.linewidth": THEME_LINE_WIDTH,
            "lines.markersize": THEME_MARKER_SIZE,
            "lines.markeredgewidth": THEME_MARKER_EDGE_WIDTH,
        }
    )
