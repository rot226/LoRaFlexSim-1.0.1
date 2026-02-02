"""Valeurs par défaut pour la taille des figures matplotlib."""

from __future__ import annotations

from typing import Tuple

from article_c.common.plotting_style import (
    DOUBLE_COLUMN_FIGSIZE,
    DOUBLE_COLUMN_WIDTH,
    HEIGHT_RATIO,
    SINGLE_COLUMN_FIGSIZE,
    SINGLE_COLUMN_WIDTH,
)

DEFAULT_FIGSIZE_SIMPLE: Tuple[float, float] = SINGLE_COLUMN_FIGSIZE
DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = DOUBLE_COLUMN_FIGSIZE

IEEE_SINGLE_COLUMN_WIDTH: float = SINGLE_COLUMN_WIDTH
IEEE_DOUBLE_COLUMN_WIDTH: float = DOUBLE_COLUMN_WIDTH
IEEE_HEIGHT_RATIO: float = HEIGHT_RATIO


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
        width = IEEE_DOUBLE_COLUMN_WIDTH
    else:
        width = IEEE_SINGLE_COLUMN_WIDTH
    return (width, width * IEEE_HEIGHT_RATIO)
