"""Valeurs par défaut pour la taille des figures matplotlib."""

from __future__ import annotations

from typing import Tuple

DEFAULT_FIGSIZE_SIMPLE: Tuple[float, float] = (6.0, 4.0)
DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = (7.0, 4.0)

IEEE_SINGLE_COLUMN_WIDTH: float = 3.5
IEEE_DOUBLE_COLUMN_WIDTH: float = 7.16
IEEE_HEIGHT_RATIO: float = 2 / 3


def resolve_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille de figure selon le nombre de séries/algorithmes."""
    if num_series and num_series > 1:
        width, height = DEFAULT_FIGSIZE_MULTI
    else:
        width, height = DEFAULT_FIGSIZE_SIMPLE
    return (width, height + 2)


def resolve_ieee_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille IEEE (simple/double colonne) selon le nombre de séries."""
    if num_series and num_series > 1:
        width = IEEE_DOUBLE_COLUMN_WIDTH
    else:
        width = IEEE_SINGLE_COLUMN_WIDTH
    return (width, width * IEEE_HEIGHT_RATIO)
