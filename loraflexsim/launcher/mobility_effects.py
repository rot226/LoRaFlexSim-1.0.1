"""Effets numériques simplifiés de mobilité.

Ce module reste volontairement indépendant d'OMNeT/FLoRa pour permettre
une utilisation dans des pipelines analytiques (tracés, calibrations, etc.).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


_MODEL_ALIASES = {
    "sm": "sm",
    "smooth": "sm",
    "smooth_mobility": "sm",
    "rwp": "rwp",
    "random_waypoint": "rwp",
    "random-waypoint": "rwp",
}


def _normalize_model(model: str) -> str:
    key = model.strip().lower()
    normalized = _MODEL_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(sorted(_MODEL_ALIASES))
        raise ValueError(f"unknown mobility model '{model}', expected one of: {supported}")
    return normalized


def mobility_penalty(model: str, speed: float) -> float:
    """Retourne une pénalité de mobilité croissante avec la vitesse (dB).

    Règles garanties:
    - monotonie croissante avec ``speed``;
    - ``rwp`` systématiquement plus pénalisant que ``sm``;
    - forme douce (linéaire + quadratique) pour éviter des ruptures.
    """

    if speed < 0:
        raise ValueError("speed must be >= 0")

    normalized = _normalize_model(model)
    if normalized == "sm":
        base, linear, quad = 0.08, 0.030, 0.0025
    else:  # rwp
        base, linear, quad = 0.16, 0.038, 0.0034

    return base + linear * speed + quad * speed * speed


def stochastic_variation(model: str, rng, scale: float) -> float:
    """Retourne un bruit aléatoire faible et borné.

    Le bruit est centré sur 0, contrôlé par ``scale``, avec un clipping à
    ``±3σ`` pour garder des courbes stables et lisses entre runs.
    """

    if scale < 0:
        raise ValueError("scale must be >= 0")

    normalized = _normalize_model(model)
    base_std = 0.012 if normalized == "sm" else 0.018
    sigma = scale * base_std
    if sigma == 0:
        return 0.0

    draw: Callable[..., float] | None = getattr(rng, "normal", None)
    if draw is None:
        draw = getattr(rng, "gauss", None)
    if draw is None:
        raise TypeError("rng must expose either normal(mean, std) or gauss(mean, std)")

    noise = float(draw(0.0, sigma))
    bound = 3.0 * sigma
    return max(-bound, min(bound, noise))


def _extract_speeds(config: dict) -> np.ndarray:
    if "speeds" in config:
        speeds = np.asarray(config["speeds"], dtype=float)
    else:
        min_speed = float(config.get("min_speed", 0.0))
        max_speed = float(config.get("max_speed", 20.0))
        points = int(config.get("num_points", 11))
        if points < 2:
            raise ValueError("num_points must be >= 2")
        speeds = np.linspace(min_speed, max_speed, points)

    if speeds.ndim != 1 or len(speeds) < 2:
        raise ValueError("speeds must be a one-dimensional array with at least 2 values")
    if np.any(speeds < 0):
        raise ValueError("speeds must be >= 0")

    return np.sort(np.unique(speeds))


def _monotone_decreasing(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values)


def _draw_noise(rng, size: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(size, dtype=float)

    draw: Callable[..., np.ndarray] | Callable[..., float] | None = getattr(rng, "normal", None)
    if draw is None:
        draw = getattr(rng, "gauss", None)
    if draw is None:
        raise TypeError("rng must expose either normal(mean, std) or gauss(mean, std)")

    try:
        samples = np.asarray(draw(0.0, sigma, size=size), dtype=float)
    except TypeError:
        samples = np.asarray([float(draw(0.0, sigma)) for _ in range(size)], dtype=float)

    return np.clip(samples, -3.0 * sigma, 3.0 * sigma)


def generate_fig1_pdr_vs_speed(config: dict, rng) -> pd.DataFrame:
    """Génère la courbe PDR(vitesse) pour SM et RWP (Fig.1)."""

    speeds = _extract_speeds(config)
    speed_norm = speeds / max(float(speeds.max()), 1.0)

    pdr0_sm = float(config.get("pdr0_sm", 0.95))
    pdr0_gap = float(config.get("pdr0_gap", 0.03))
    slope_sm = float(config.get("pdr_slope_sm", 0.18))
    slope_gap = float(config.get("pdr_slope_gap", 0.06))
    noise_sigma = float(config.get("pdr_noise_sigma", 0.01))

    base_sm = pdr0_sm - slope_sm * speed_norm - 0.04 * speed_norm * speed_norm
    base_rwp = (pdr0_sm - pdr0_gap) - (slope_sm + slope_gap) * speed_norm - 0.05 * speed_norm * speed_norm

    sm = _monotone_decreasing(base_sm + _draw_noise(rng, len(speeds), noise_sigma))
    rwp = _monotone_decreasing(base_rwp + _draw_noise(rng, len(speeds), noise_sigma))
    rwp = np.minimum(rwp, sm - 1e-3)

    sm = np.clip(sm, 0.0, 1.0)
    rwp = np.clip(rwp, 0.0, 1.0)

    return pd.DataFrame({"speed": speeds, "pdr_sm": sm, "pdr_rwp": rwp})


def generate_fig4_der_vs_speed(config: dict, rng) -> pd.DataFrame:
    """Génère la courbe DER(vitesse) pour SM et RWP (Fig.4)."""

    speeds = _extract_speeds(config)
    speed_norm = speeds / max(float(speeds.max()), 1.0)

    der0_sm = float(config.get("der0_sm", 0.88))
    der0_gap = float(config.get("der0_gap", 0.04))
    slope_sm = float(config.get("der_slope_sm", 0.22))
    slope_gap = float(config.get("der_slope_gap", 0.08))
    noise_sigma = float(config.get("der_noise_sigma", 0.012))

    base_sm = der0_sm - slope_sm * speed_norm - 0.05 * speed_norm * speed_norm
    base_rwp = (der0_sm - der0_gap) - (slope_sm + slope_gap) * speed_norm - 0.06 * speed_norm * speed_norm

    sm = _monotone_decreasing(base_sm + _draw_noise(rng, len(speeds), noise_sigma))
    rwp = _monotone_decreasing(base_rwp + _draw_noise(rng, len(speeds), noise_sigma))
    rwp = np.minimum(rwp, sm - 1e-3)

    sm = np.clip(sm, 0.0, 1.0)
    rwp = np.clip(rwp, 0.0, 1.0)

    return pd.DataFrame({"speed": speeds, "der_sm": sm, "der_rwp": rwp})
