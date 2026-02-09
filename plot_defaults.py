"""Valeurs par défaut pour la taille des figures matplotlib."""

from __future__ import annotations

from typing import Tuple

from article_c.common.plotting_style import (
    DOUBLE_COLUMN_FIGSIZE,
    DOUBLE_COLUMN_WIDTH,
    SINGLE_COLUMN_FIGSIZE,
    SINGLE_COLUMN_WIDTH,
)

DEFAULT_FIGSIZE_SIMPLE: Tuple[float, float] = SINGLE_COLUMN_FIGSIZE
DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = DOUBLE_COLUMN_FIGSIZE

IEEE_SINGLE_COLUMN_WIDTH: float = SINGLE_COLUMN_WIDTH
IEEE_DOUBLE_COLUMN_WIDTH: float = DOUBLE_COLUMN_WIDTH
IEEE_HEIGHT_RATIO: float = SINGLE_COLUMN_FIGSIZE[1] / SINGLE_COLUMN_FIGSIZE[0]
RL_FIGURE_SCALE: float = 1.15
WIDE_SERIES_THRESHOLD: int = 3
WIDE_SERIES_WIDTH_SCALE: float = 1.12
WIDE_SERIES_WSPACE: float = 0.32


def resolve_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille de figure selon le nombre de séries/algorithmes."""
    if num_series and num_series > 1:
        width, height = DEFAULT_FIGSIZE_MULTI
    else:
        width, height = DEFAULT_FIGSIZE_SIMPLE
    return (width, height)


def resolve_ieee_figsize(
    num_series: int | None = None,
    *,
    scale: float = 1.0,
) -> Tuple[float, float]:
    """Retourne la taille IEEE (simple/double colonne) selon le nombre de séries."""
    if scale <= 0:
        raise ValueError("scale doit être strictement positif.")
    if num_series and num_series > 1:
        return (
            IEEE_DOUBLE_COLUMN_WIDTH * scale,
            IEEE_DOUBLE_COLUMN_WIDTH * IEEE_HEIGHT_RATIO * scale,
        )
    return (
        IEEE_SINGLE_COLUMN_WIDTH * scale,
        IEEE_SINGLE_COLUMN_WIDTH * IEEE_HEIGHT_RATIO * scale,
    )


def resolve_ieee_figsize_for_series(
    num_series: int | None = None,
    *,
    scale: float = 1.0,
    wide_series_threshold: int = WIDE_SERIES_THRESHOLD,
    wide_width_scale: float = WIDE_SERIES_WIDTH_SCALE,
) -> Tuple[float, float]:
    """Retourne la taille IEEE en élargissant si plusieurs séries sont superposées."""
    width, height = resolve_ieee_figsize(num_series, scale=scale)
    if num_series and num_series >= wide_series_threshold:
        width *= wide_width_scale
    return (width, height)
