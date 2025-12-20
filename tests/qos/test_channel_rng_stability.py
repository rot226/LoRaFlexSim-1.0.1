from __future__ import annotations

import math
from typing import Iterable, List

from loraflexsim.launcher.qos import QoSManager
from loraflexsim.scenarios.qos_cluster_bench import _create_simulator


def _snir_series(*, use_snir: bool, seed: int) -> List[float]:
    """Exécute une courte simulation et retourne les SNIR moyens par paquet."""

    simulator = _create_simulator(
        8,
        3.0,
        seed,
        use_snir=use_snir,
    )
    QoSManager().apply(simulator, "ADR-Pure", use_snir=use_snir)
    simulator.run(max_time=20.0)

    values: list[float] = []
    for entry in simulator.events_log:
        if not entry.get("heard"):
            continue
        snir = entry.get("snir_dB")
        if snir is None or not math.isfinite(snir):
            snir = entry.get("snr_dB")
        if snir is None or not math.isfinite(snir):
            continue
        values.append(float(snir))

    assert values, "Aucun échantillon SNIR collecté"
    return values


def _assert_same_sequence(series: Iterable[float], reference: Iterable[float]) -> None:
    assert list(series) == list(reference), "Les séries SNIR devraient être identiques"


def test_channel_rng_stability_and_independence() -> None:
    series_on_first = _snir_series(use_snir=True, seed=21)
    series_on_second = _snir_series(use_snir=True, seed=21)
    _assert_same_sequence(series_on_second, series_on_first)

    series_off_first = _snir_series(use_snir=False, seed=21)
    series_off_second = _snir_series(use_snir=False, seed=21)
    _assert_same_sequence(series_off_second, series_off_first)

    changed_seed_series = _snir_series(use_snir=True, seed=37)
    assert changed_seed_series != series_on_first, "Les séries doivent diverger quand la graine change"
