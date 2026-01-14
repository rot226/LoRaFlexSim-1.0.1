"""Outils simples pour l'interférence et le SNIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math


@dataclass(frozen=True)
class Signal:
    """Représente un signal reçu."""

    rssi_dbm: float
    sf: int
    channel_hz: int


@dataclass(frozen=True)
class InterferenceOutcome:
    """Résumé d'un calcul d'interférence et de réception."""

    success: bool
    outage: bool
    snir_db: float | None
    interference_dbm: float
    co_sf_collisions: int


def aggregate_interference(powers_dbm: Iterable[float]) -> float:
    """Agrège des puissances en dBm via une somme linéaire simplifiée."""

    linear = [10 ** (p / 10) for p in powers_dbm]
    total_linear = sum(linear)
    return 10 * math.log10(total_linear) if total_linear > 0 else float("-inf")


def co_sf_interferers(target: Signal, interferers: Iterable[Signal]) -> list[Signal]:
    """Retourne les interférences co-SF sur le même canal que ``target``."""

    return [
        interferer
        for interferer in interferers
        if interferer.sf == target.sf and interferer.channel_hz == target.channel_hz
    ]


def compute_snir_db(
    signal_dbm: float,
    interferers_dbm: Sequence[float],
    noise_dbm: float,
) -> float:
    """Calcule le SNIR (dB) à partir du signal, des interférences et du bruit."""

    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interferers_dbm)
    noise_linear = 10 ** (noise_dbm / 10)
    denominator = interference_linear + noise_linear
    if denominator <= 0:
        return float("inf")
    return 10 * math.log10(signal_linear / denominator)


def evaluate_reception(
    target: Signal,
    interferers: Iterable[Signal],
    *,
    sensitivity_dbm: float,
    snir_enabled: bool = True,
    capture_threshold_db: float = 1.0,
    noise_dbm: float = -174.0,
) -> InterferenceOutcome:
    """Évalue la réception avec ou sans SNIR et détecte un outage.

    - SNIR OFF: le succès dépend uniquement de ``target.rssi_dbm`` >= sensibilité.
    - SNIR ON: le SNIR est calculé avec la somme des interférences co-SF.
    """

    co_sf = co_sf_interferers(target, interferers)
    interference_dbm = aggregate_interference([entry.rssi_dbm for entry in co_sf])
    rssi_ok = target.rssi_dbm >= sensitivity_dbm

    snir_db = None
    snir_ok = True
    if snir_enabled:
        snir_db = compute_snir_db(
            signal_dbm=target.rssi_dbm,
            interferers_dbm=[entry.rssi_dbm for entry in co_sf],
            noise_dbm=noise_dbm,
        )
        snir_ok = snir_db >= capture_threshold_db

    success = rssi_ok if not snir_enabled else (rssi_ok and snir_ok)
    outage = (not rssi_ok) or (snir_enabled and not snir_ok)

    return InterferenceOutcome(
        success=success,
        outage=outage,
        snir_db=snir_db,
        interference_dbm=interference_dbm,
        co_sf_collisions=len(co_sf),
    )
