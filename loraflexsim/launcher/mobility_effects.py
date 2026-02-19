"""Effets numériques simplifiés de mobilité.

Ce module reste volontairement indépendant d'OMNeT/FLoRa pour permettre
une utilisation dans des pipelines analytiques (tracés, calibrations, etc.).
"""

from __future__ import annotations

from collections.abc import Callable


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
