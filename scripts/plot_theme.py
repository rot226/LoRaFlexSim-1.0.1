"""Thème matplotlib partagé pour les scripts de visualisation."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

try:
    from article_c.common.plotting_style import apply_base_rcparams

    _ARTICLE_C_STYLE_AVAILABLE = True
except Exception:  # pragma: no cover - dépend de l'environnement d'exécution
    apply_base_rcparams = None  # type: ignore[assignment]
    _ARTICLE_C_STYLE_AVAILABLE = False

_FALLBACK_WARNING_EMITTED = False

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
    if _ARTICLE_C_STYLE_AVAILABLE and apply_base_rcparams is not None:
        apply_base_rcparams()
    else:
        global _FALLBACK_WARNING_EMITTED
        if not _FALLBACK_WARNING_EMITTED:
            warnings.warn(
                "article_c indisponible: utilisation du style matplotlib par défaut.",
                RuntimeWarning,
                stacklevel=2,
            )
            _FALLBACK_WARNING_EMITTED = True
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
