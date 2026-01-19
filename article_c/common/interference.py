"""Outils simples pour l'interférence et le SNIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import heapq
import math
import random
from collections import defaultdict

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


@dataclass(frozen=True)
class InterferenceSweepResult:
    """Résultat d'un sweep line pour les overlaps co-SF."""

    overlaps_by_index: list[list[int]]
    load_counter: int


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
    """Retourne les interférences co-SF (même canal) en overlap temporel."""

    return [
        interferer
        for interferer in interferers
        if (
            interferer.sf == target.sf
            and interferer.channel_hz == target.channel_hz
            and _signals_overlap(target, interferer)
        )
    ]


def compute_co_sf_overlaps(signals: Sequence[Signal]) -> InterferenceSweepResult:
    """Construit les overlaps co-SF avec un sweep line (fenêtre glissante).

    Les signaux sont triés par canal, SF et temps de départ. Le compteur de charge
    estime le nombre de comparaisons effectuées.
    """

    overlaps: list[list[int]] = [[] for _ in signals]
    load_counter = 0
    grouped: dict[tuple[int, int], list[tuple[int, Signal]]] = defaultdict(list)
    for idx, signal in enumerate(signals):
        grouped[(signal.channel_hz, signal.sf)].append((idx, signal))

    for entries in grouped.values():
        known: list[tuple[int, Signal]] = []
        unknown: list[tuple[int, Signal]] = []
        for idx, signal in entries:
            if signal.start_time_s is None or signal.end_time_s is None:
                unknown.append((idx, signal))
            else:
                known.append((idx, signal))

        known.sort(key=lambda item: (item[1].start_time_s, item[1].end_time_s))
        active_heap: list[tuple[float, int]] = []
        active_set: set[int] = set()
        for idx, signal in known:
            start_time = signal.start_time_s
            end_time = signal.end_time_s
            if start_time is None or end_time is None:
                continue
            while active_heap and active_heap[0][0] <= start_time:
                _, expired_idx = heapq.heappop(active_heap)
                active_set.discard(expired_idx)
            for other_idx in active_set:
                load_counter += 1
                overlaps[idx].append(other_idx)
                overlaps[other_idx].append(idx)
            heapq.heappush(active_heap, (end_time, idx))
            active_set.add(idx)

        if unknown:
            known_indices = [idx for idx, _ in known]
            unknown_indices = [idx for idx, _ in unknown]
            for unknown_idx in unknown_indices:
                for other_idx in known_indices:
                    load_counter += 1
                    overlaps[unknown_idx].append(other_idx)
                    overlaps[other_idx].append(unknown_idx)
            for index, first_idx in enumerate(unknown_indices):
                for second_idx in unknown_indices[index + 1 :]:
                    load_counter += 1
                    overlaps[first_idx].append(second_idx)
                    overlaps[second_idx].append(first_idx)

    return InterferenceSweepResult(
        overlaps_by_index=overlaps,
        load_counter=load_counter,
    )


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


def compute_sir_db(
    signal_dbm: float,
    interferers_dbm: Sequence[float],
) -> float:
    """Calcule le SIR (dB) à partir du signal et des interférences."""

    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interferers_dbm)
    if interference_linear <= 0:
        return float("inf")
    return 10 * math.log10(signal_linear / interference_linear)


def compute_thermal_noise_dbm(
    noise_floor_dbm: float,
    bandwidth_hz: float | None,
) -> float:
    """Calcule le bruit thermique total (dBm) à partir d'une densité de bruit."""

    if bandwidth_hz is None or bandwidth_hz <= 0:
        return noise_floor_dbm
    return noise_floor_dbm + 10 * math.log10(bandwidth_hz)


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
        start_time_s=signal.start_time_s,
        end_time_s=signal.end_time_s,
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
    snir_threshold_db: float = 5.0,
    noise_floor_dbm: float = -174.0,
    bandwidth_hz: float | None = 125_000.0,
) -> InterferenceOutcome:
    """Évalue la réception avec ou sans SNIR et détecte un outage.

    - SNIR OFF: le succès dépend uniquement de ``target.rssi_dbm`` >= sensibilité.
    - SNIR ON: succès si RSSI >= sensibilité ET SNIR >= seuil (bruit thermique inclus).
    - Capture effect: en co-SF, succès seulement si SIR >= seuil.
    """

    co_sf = co_sf_interferers(target, interferers)
    interferer_powers_dbm = [entry.rssi_dbm for entry in co_sf]
    interference_dbm = aggregate_interference(interferer_powers_dbm)
    thermal_noise_dbm = compute_thermal_noise_dbm(noise_floor_dbm, bandwidth_hz)
    rssi_ok = target.rssi_dbm >= sensitivity_dbm
    effective_snir_threshold_db = min(max(snir_threshold_db, 3.0), 6.0)

    snir_db = None
    snir_ok = True
    if snir_enabled:
        snir_db = compute_snir_db(
            signal_dbm=target.rssi_dbm,
            interferers_dbm=interferer_powers_dbm,
            noise_dbm=thermal_noise_dbm,
        )
        snir_ok = snir_db >= effective_snir_threshold_db

    capture_ok = True
    if co_sf:
        sir_db = compute_sir_db(
            signal_dbm=target.rssi_dbm,
            interferers_dbm=interferer_powers_dbm,
        )
        capture_ok = sir_db >= effective_snir_threshold_db

    success = (
        rssi_ok
        if not snir_enabled
        else (rssi_ok and snir_ok)
    )
    success = success and capture_ok
    outage = (not rssi_ok) or (snir_enabled and (not snir_ok)) or (not capture_ok)

    return InterferenceOutcome(
        success=success,
        outage=outage,
        snir_db=snir_db,
        interference_dbm=interference_dbm,
        co_sf_collisions=len(co_sf),
    )
