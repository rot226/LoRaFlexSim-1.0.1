"""Modèles de propagation et d'atténuation simplifiés."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Tuple


def log_distance_path_loss(
    distance_m: float,
    freq_mhz: float,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 1.0,
    shadowing_std_db: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """Retourne la perte (dB) via un modèle log-distance avec fading optionnel.

    Hypothèses :
    - La perte de référence est celle de l'espace libre à ``reference_distance_m``.
    - Le fading est un bruit log-normal (en dB) de moyenne 0 et écart-type
      ``shadowing_std_db``.
    """

    if distance_m <= 0:
        raise ValueError("distance_m doit être positive")
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m doit être positive")
    if freq_mhz <= 0:
        raise ValueError("freq_mhz doit être positive")
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent doit être positive")

    fspl_db = 32.45 + 20 * math.log10(reference_distance_m / 1000) + 20 * math.log10(freq_mhz)
    path_loss_db = fspl_db + 10 * path_loss_exponent * math.log10(distance_m / reference_distance_m)
    if shadowing_std_db > 0:
        generator = rng or random
        path_loss_db += generator.gauss(0.0, shadowing_std_db)
    return path_loss_db


def free_space_path_loss(distance_km: float, freq_mhz: float) -> float:
    """Retourne une perte en dB via le modèle de l'espace libre.

    Hypothèse : propagation en champ libre (pas de fading).
    """

    if distance_km <= 0:
        raise ValueError("distance_km doit être positive")
    if freq_mhz <= 0:
        raise ValueError("freq_mhz doit être positive")
    return 32.45 + 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz)


def generate_positions_in_disk(
    count: int,
    radius_m: float,
    rng: random.Random | None = None,
) -> List[Tuple[float, float]]:
    """Génère des positions uniformes dans un disque de rayon ``radius_m``.

    Hypothèses :
    - Distribution uniforme en surface (rayon tiré avec racine de U[0,1]).
    - Les positions sont centrées en (0, 0).
    """

    if count < 0:
        raise ValueError("count doit être positif ou nul")
    if radius_m <= 0:
        raise ValueError("radius_m doit être positif")

    generator = rng or random
    positions: List[Tuple[float, float]] = []
    for _ in range(count):
        u = generator.random()
        v = generator.random()
        r = math.sqrt(u) * radius_m
        theta = 2 * math.pi * v
        positions.append((r * math.cos(theta), r * math.sin(theta)))
    return positions


def rssi_dbm(ptx_dbm: float, path_loss_db: float) -> float:
    """Calcule le RSSI (dBm) à partir de la puissance émise et de la perte."""

    return ptx_dbm - path_loss_db


def rssi_for_positions(
    positions: Iterable[Tuple[float, float]],
    ptx_dbm: float,
    freq_mhz: float,
    path_loss_exponent: float = 2.7,
    shadowing_std_db: float = 0.0,
    rng: random.Random | None = None,
) -> List[float]:
    """Calcule un RSSI pour chaque position en supposant une station en (0,0).

    Hypothèses :
    - Distance calculée au centre du disque (0, 0).
    - Modèle log-distance avec fading log-normal optionnel.
    """

    rssis: List[float] = []
    for x, y in positions:
        distance = math.hypot(x, y)
        loss_db = log_distance_path_loss(
            distance_m=distance,
            freq_mhz=freq_mhz,
            path_loss_exponent=path_loss_exponent,
            shadowing_std_db=shadowing_std_db,
            rng=rng,
        )
        rssis.append(rssi_dbm(ptx_dbm, loss_db))
    return rssis
