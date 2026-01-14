"""Outils simples pour l'interférence et le SNIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import random

from article_c.common.propagation import sample_fading_db


@dataclass(frozen=True)
class Signal:
    """Représente un signal reçu."""

    rssi_dbm: float
    sf: int
    channel_hz: int
    start_time_s: float | None = None
    end_time_s: float | None = None


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


def _signals_overlap(first: Signal, second: Signal) -> bool:
    """Indique si deux signaux se chevauchent temporellement."""

    if (
        first.start_time_s is None
        or first.end_time_s is None
        or second.start_time_s is None
        or second.end_time_s is None
    ):
        return True
    return not (
        second.end_time_s <= first.start_time_s
        or second.start_time_s >= first.end_time_s
    )


def co_sf_interferers(target: Signal, interferers: Iterable[Signal]) -> list[Signal]:
    """Retourne les interférences co-SF sur le même canal et en overlap temporel."""

    return [
        interferer
        for interferer in interferers
        if (
            interferer.sf == target.sf
            and interferer.channel_hz == target.channel_hz
            and _signals_overlap(target, interferer)
        )
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


def apply_fading_to_signal(
    signal: Signal,
    *,
    fading_type: str | None = None,
    fading_sigma_db: float = 0.0,
    fading_mean_db: float = 0.0,
    rng: random.Random | None = None,
) -> Signal:
    """Applique un fading en dB au RSSI d'un signal."""

    generator = rng or random
    fading_db = sample_fading_db(
        fading_type,
        sigma_db=fading_sigma_db,
        mean_db=fading_mean_db,
        rng=generator,
    )
    return Signal(
        rssi_dbm=signal.rssi_dbm - fading_db,
        sf=signal.sf,
        channel_hz=signal.channel_hz,
    )


def apply_fading_to_signals(
    signals: Iterable[Signal],
    *,
    fading_type: str | None = None,
    fading_sigma_db: float = 0.0,
    fading_mean_db: float = 0.0,
    rng: random.Random | None = None,
) -> list[Signal]:
    """Applique un fading à une liste de signaux."""

    generator = rng or random
    return [
        apply_fading_to_signal(
            signal,
            fading_type=fading_type,
            fading_sigma_db=fading_sigma_db,
            fading_mean_db=fading_mean_db,
            rng=generator,
        )
        for signal in signals
    ]


def evaluate_reception(
    target: Signal,
    interferers: Iterable[Signal],
    *,
    sensitivity_dbm: float,
    snir_enabled: bool = True,
    snir_threshold_db: float = 1.0,
    noise_floor_dbm: float = -174.0,
) -> InterferenceOutcome:
    """Évalue la réception avec ou sans SNIR et détecte un outage.

    - SNIR OFF: le succès dépend uniquement de ``target.rssi_dbm`` >= sensibilité.
    - SNIR ON: le SNIR est calculé avec la somme des interférences co-SF.
    """

    co_sf = co_sf_interferers(target, interferers)
    interferer_powers_dbm = [entry.rssi_dbm for entry in co_sf]
    interference_dbm = aggregate_interference(interferer_powers_dbm)
    effective_rssi_dbm = aggregate_interference([target.rssi_dbm, noise_floor_dbm])
    rssi_ok = effective_rssi_dbm >= sensitivity_dbm

    snir_db = None
    snir_ok = True
    capture_ok = True
    if snir_enabled:
        snir_db = compute_snir_db(
            signal_dbm=target.rssi_dbm,
            interferers_dbm=interferer_powers_dbm,
            noise_dbm=noise_floor_dbm,
        )
        snir_ok = snir_db >= snir_threshold_db
        if interferer_powers_dbm:
            capture_margin_db = target.rssi_dbm - max(interferer_powers_dbm)
            capture_ok = capture_margin_db >= snir_threshold_db

    success = (
        rssi_ok
        if not snir_enabled
        else (rssi_ok and snir_ok and capture_ok)
    )
    outage = (not rssi_ok) or (snir_enabled and (not snir_ok or not capture_ok))

    return InterferenceOutcome(
        success=success,
        outage=outage,
        snir_db=snir_db,
        interference_dbm=interference_dbm,
        co_sf_collisions=len(co_sf),
    )
