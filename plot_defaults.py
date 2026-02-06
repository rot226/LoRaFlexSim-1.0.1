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
IEEE_MAX_FIGSIZE: Tuple[float, float] = (8.0, 6.0)


def _cap_figsize(figsize: Tuple[float, float]) -> Tuple[float, float]:
    max_width, max_height = IEEE_MAX_FIGSIZE
    width, height = figsize
    if width <= 0 or height <= 0:
        return figsize
    scale = min(max_width / width, max_height / height, 1.0)
    return (width * scale, height * scale)


def resolve_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille de figure selon le nombre de séries/algorithmes."""
    if num_series and num_series > 1:
        width, height = DEFAULT_FIGSIZE_MULTI
    else:
        width, height = DEFAULT_FIGSIZE_SIMPLE
    return (width, height)


def resolve_ieee_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille IEEE (simple/double colonne) selon le nombre de séries."""
    if num_series and num_series > 1:
        return _cap_figsize(
            (
                IEEE_DOUBLE_COLUMN_WIDTH,
                IEEE_DOUBLE_COLUMN_WIDTH * IEEE_HEIGHT_RATIO,
            )
        )
    return _cap_figsize(
        (
            IEEE_SINGLE_COLUMN_WIDTH,
            IEEE_SINGLE_COLUMN_WIDTH * IEEE_HEIGHT_RATIO,
        )
    )
