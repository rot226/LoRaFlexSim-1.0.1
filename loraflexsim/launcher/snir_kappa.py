"""Helpers to weight inter-SF interference with kappa coefficients."""

from __future__ import annotations

from typing import Mapping


DEFAULT_SF_MIN = 7
DEFAULT_SF_MAX = 12


def default_kappa_matrix(
    alpha_isf: float = 0.0, *, sf_min: int = DEFAULT_SF_MIN, sf_max: int = DEFAULT_SF_MAX
) -> list[list[float]]:
    """Return a square kappa matrix with ``alpha_isf`` off-diagonal values."""

    alpha = float(alpha_isf)
    matrix: list[list[float]] = []
    for sf in range(sf_min, sf_max + 1):
        row: list[float] = []
        for sfk in range(sf_min, sf_max + 1):
            row.append(1.0 if sf == sfk else alpha)
        matrix.append(row)
    return matrix


def kappa_factor(
    target_sf: int | None,
    interferer_sf: int | None,
    *,
    alpha_isf: float = 0.0,
    kappa_isf: object | None = None,
    sf_min: int = DEFAULT_SF_MIN,
) -> float:
    """Return the kappa coefficient for ``target_sf`` vs ``interferer_sf``."""

    if target_sf is None or interferer_sf is None:
        return 1.0
    if target_sf == interferer_sf:
        return 1.0
    if kappa_isf is None:
        return float(alpha_isf)

    if isinstance(kappa_isf, Mapping):
        row = kappa_isf.get(target_sf)
        if isinstance(row, Mapping):
            value = row.get(interferer_sf)
            if value is not None:
                return float(value)
        if isinstance(row, (list, tuple)):
            idx = interferer_sf - sf_min
            if 0 <= idx < len(row):
                return float(row[idx])

    if isinstance(kappa_isf, (list, tuple)):
        idx = target_sf - sf_min
        if 0 <= idx < len(kappa_isf):
            row = kappa_isf[idx]
            if isinstance(row, (list, tuple)):
                jdx = interferer_sf - sf_min
                if 0 <= jdx < len(row):
                    return float(row[jdx])

    return float(alpha_isf)
