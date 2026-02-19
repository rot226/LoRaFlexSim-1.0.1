"""Générateurs de jeux de données synthétiques pour les figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _get_int(config: Mapping[str, Any], *keys: str, default: int) -> int:
    """Retourne la première valeur entière trouvée pour les clés demandées."""
    for key in keys:
        if key in config:
            return int(config[key])
    return default


def _get_float(config: Mapping[str, Any], *keys: str, default: float) -> float:
    """Retourne la première valeur flottante trouvée pour les clés demandées."""
    for key in keys:
        if key in config:
            return float(config[key])
    return default


def generate_fig2_learning_curve(config: Mapping[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Génère une courbe d'apprentissage stable pour la figure 2.

    Parameters
    ----------
    config:
        Mapping de configuration. Les clés suivantes sont optionnelles :
        - ``episodes`` / ``n_episodes`` / ``fig2_episodes`` (int)
        - ``fig2_noise_std`` / ``noise_std`` (float)
        - ``fig2_start_reward`` / ``start_reward`` (float)
    rng:
        Générateur pseudo-aléatoire NumPy pour une reproductibilité totale.

    Returns
    -------
    pandas.DataFrame
        Colonnes: ``episode``, ``reward_v1``, ``reward_v5``, ``reward_v10``.
    """

    episodes = _get_int(config, "fig2_episodes", "n_episodes", "episodes", default=250)
    noise_std = _get_float(config, "fig2_noise_std", "noise_std", default=0.004)
    start_reward = _get_float(config, "fig2_start_reward", "start_reward", default=0.38)

    x = np.arange(1, episodes + 1, dtype=float)

    # Exponentielle saturante : start + (plateau - start) * (1 - exp(-k * x))
    # v=1 : montée la plus rapide et plateau le plus haut.
    curve_specs = {
        "reward_v1": {"plateau": 0.94, "k": 0.055},
        "reward_v5": {"plateau": 0.90, "k": 0.040},
        "reward_v10": {"plateau": 0.87, "k": 0.028},
    }

    data: dict[str, np.ndarray] = {"episode": x.astype(int)}
    for name, spec in curve_specs.items():
        deterministic = start_reward + (spec["plateau"] - start_reward) * (1.0 - np.exp(-spec["k"] * x))

        # Bruit très léger, un peu plus marqué au début pour l'effet expérimental.
        noise_scale = noise_std * (0.7 + 0.3 * np.exp(-x / 80.0))
        noisy = deterministic + rng.normal(loc=0.0, scale=noise_scale, size=episodes)

        # Encadrement pour garder des courbes réalistes et stables.
        data[name] = np.clip(noisy, 0.0, 1.0)

    return pd.DataFrame(data)
