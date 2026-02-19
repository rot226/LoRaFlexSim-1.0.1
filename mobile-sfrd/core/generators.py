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


def generate_fig3_sf_hist(config: Mapping[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Génère les histogrammes de SF pour la figure 3.

    Parameters
    ----------
    config:
        Mapping de configuration. Les clés suivantes sont optionnelles :
        - ``fig3_nodes`` / ``nodes_count`` (int)
        - ``fig3_sf_min`` (int)
        - ``fig3_sf_max`` (int)
    rng:
        Générateur pseudo-aléatoire NumPy pour une reproductibilité totale.

    Returns
    -------
    pandas.DataFrame
        Colonnes: ``mobility``, ``speed``, ``window``, ``sf``, ``nodes_count``.
    """

    nodes_total = _get_int(config, "fig3_nodes", "nodes_count", default=200)
    sf_min = _get_int(config, "fig3_sf_min", default=7)
    sf_max = _get_int(config, "fig3_sf_max", default=12)

    if sf_max <= sf_min:
        raise ValueError("fig3_sf_max doit être strictement supérieur à fig3_sf_min.")

    sf_values = np.arange(sf_min, sf_max + 1)
    n_sf = len(sf_values)

    # Distribution de départ légèrement biaisée vers les SF bas.
    initial_base = np.linspace(1.35, 0.75, n_sf)
    initial_probs = initial_base / initial_base.sum()

    rows: list[dict[str, Any]] = []
    panel_specs = [("SM", 1), ("SM", 10), ("RWP", 1), ("RWP", 10)]

    for mobility, speed in panel_specs:
        initial_counts = rng.multinomial(nodes_total, initial_probs)

        speed_norm = max(0.0, (float(speed) - 1.0) / 9.0)
        mobility_boost = 0.0 if mobility == "SM" else 0.35

        # Le tilt augmente la probabilité des SF élevés ; l'effet est plus fort
        # quand la vitesse augmente, et encore plus en RWP.
        tilt_strength = 0.10 + 0.35 * speed_norm + mobility_boost
        tilt = np.linspace(-1.0, 1.0, n_sf)
        final_logits = np.log(initial_probs) + tilt_strength * tilt
        final_probs = np.exp(final_logits - np.max(final_logits))
        final_probs = final_probs / final_probs.sum()
        final_counts = rng.multinomial(nodes_total, final_probs)

        for window, counts in (("initial", initial_counts), ("final", final_counts)):
            for sf, count in zip(sf_values, counts):
                rows.append(
                    {
                        "mobility": mobility,
                        "speed": int(speed),
                        "window": window,
                        "sf": int(sf),
                        "nodes_count": int(count),
                    }
                )

    return pd.DataFrame(rows)
