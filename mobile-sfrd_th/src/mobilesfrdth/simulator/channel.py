"""Modèles de canal radio pour le moteur de simulation mobilesfrdth."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class ChannelConfig:
    """Paramètres des pertes radio.

    Attributes
    ----------
    reference_distance_m:
        Distance de référence ``d0`` (m).
    pathloss_at_reference_db:
        Perte à ``d0`` (dB).
    pathloss_exponent:
        Exposant log-distance ``n``.
    sigma_shadowing:
        Ecart-type du shadowing log-normal (dB).
    rayleigh_fading:
        Active un fading Rayleigh multiplicatif en puissance.
    min_distance_m:
        Distance minimale pour éviter les singularités numériques.
    """

    reference_distance_m: float = 1.0
    pathloss_at_reference_db: float = 40.0
    pathloss_exponent: float = 2.7
    sigma_shadowing: float = 4.0
    rayleigh_fading: bool = False
    min_distance_m: float = 1.0


def pathloss_log_distance_db(distance_m: float, cfg: ChannelConfig) -> float:
    """Calcule le pathloss (dB) via le modèle log-distance."""

    d = max(distance_m, cfg.min_distance_m)
    d0 = max(cfg.reference_distance_m, 1e-9)
    return cfg.pathloss_at_reference_db + 10.0 * cfg.pathloss_exponent * math.log10(d / d0)


def shadowing_lognormal_db(cfg: ChannelConfig, rng: random.Random | None = None) -> float:
    """Retourne une perturbation de shadowing gaussienne (dB)."""

    generator = rng or random
    if cfg.sigma_shadowing <= 0:
        return 0.0
    return generator.gauss(0.0, cfg.sigma_shadowing)


def rayleigh_fading_db(rng: random.Random | None = None) -> float:
    """Retourne un fading Rayleigh converti en dB (sur la puissance)."""

    generator = rng or random
    u = max(generator.random(), 1e-12)
    power_linear = -math.log(u)  # Exp(1), donc puissance Rayleigh normalisée
    return 10.0 * math.log10(power_linear)


def received_power_dbm(
    tx_power_dbm: float,
    distance_m: float,
    cfg: ChannelConfig,
    *,
    rng: random.Random | None = None,
) -> float:
    """Calcule la puissance reçue en dBm avec pathloss + shadowing + fading optionnel."""

    pl_db = pathloss_log_distance_db(distance_m=distance_m, cfg=cfg)
    shadow_db = shadowing_lognormal_db(cfg=cfg, rng=rng)
    fading_db = rayleigh_fading_db(rng=rng) if cfg.rayleigh_fading else 0.0
    return tx_power_dbm - pl_db + shadow_db + fading_db
