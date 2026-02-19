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


def generate_fig5_changepoint(config: Mapping[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Génère une série temporelle PDR avec changement de régime pour la figure 5.

    Parameters
    ----------
    config:
        Mapping de configuration. Les clés suivantes sont optionnelles :
        - ``fig5_points`` / ``n_points`` / ``points`` (int)
        - ``fig5_changepoint_t`` / ``changepoint_t`` (int)
        - ``fig5_initial_pdr`` / ``initial_pdr`` (float)
        - ``fig5_drop_magnitude`` / ``drop_magnitude`` (float)
        - ``fig5_final_gap`` / ``final_gap`` (float)
        - ``fig5_noise_std`` / ``noise_std`` (float)
    rng:
        Générateur pseudo-aléatoire NumPy pour une reproductibilité totale.

    Returns
    -------
    pandas.DataFrame
        Colonnes: ``t``, ``pdr``, ``changepoint_t``.
    """

    n_points = _get_int(config, "fig5_points", "n_points", "points", default=220)
    changepoint_t = _get_int(config, "fig5_changepoint_t", "changepoint_t", default=90)
    initial_pdr = _get_float(config, "fig5_initial_pdr", "initial_pdr", default=0.92)
    drop_magnitude = _get_float(config, "fig5_drop_magnitude", "drop_magnitude", default=0.26)
    final_gap = _get_float(config, "fig5_final_gap", "final_gap", default=0.08)
    noise_std = _get_float(config, "fig5_noise_std", "noise_std", default=0.01)

    if n_points < 3:
        raise ValueError("Le nombre de points doit être au moins égal à 3.")
    if not (1 <= changepoint_t < n_points):
        raise ValueError("changepoint_t doit être dans l'intervalle [1, n_points-1].")

    t = np.arange(n_points, dtype=int)

    # Pré-changement : plateau quasi stable avec une légère oscillation.
    pre_trend = initial_pdr + 0.004 * np.sin(2.0 * np.pi * t / max(20.0, n_points / 6.0))

    # Changement brutal puis récupération incomplète vers un nouveau plateau plus bas.
    dropped_level = initial_pdr - abs(drop_magnitude)
    final_level = initial_pdr - abs(final_gap)
    if final_level > initial_pdr - 0.015:
        final_level = initial_pdr - 0.015
    if final_level <= dropped_level:
        final_level = dropped_level + 0.03

    recovery_tau = max(8.0, (n_points - changepoint_t) / 4.0)
    after_cp = final_level - (final_level - dropped_level) * np.exp(-(t - changepoint_t) / recovery_tau)

    pdr = np.where(t < changepoint_t, pre_trend, after_cp)

    # Bruit faible pour conserver la lisibilité du point de rupture.
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
    pdr = np.clip(pdr + noise, 0.0, 1.0)

    return pd.DataFrame({"t": t, "pdr": pdr, "changepoint_t": changepoint_t})
