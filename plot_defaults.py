"""Valeurs par défaut pour la taille des figures matplotlib."""

from __future__ import annotations

from typing import Tuple

DEFAULT_FIGSIZE_SIMPLE: Tuple[float, float] = (6.0, 4.0)
DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = (7.0, 4.0)


def resolve_figsize(num_series: int | None = None) -> Tuple[float, float]:
    """Retourne la taille de figure selon le nombre de séries/algorithmes."""
    if num_series and num_series > 1:
        return DEFAULT_FIGSIZE_MULTI
    return DEFAULT_FIGSIZE_SIMPLE
