from __future__ import annotations

import math
import statistics
from typing import Iterable, List

from loraflexsim.launcher.qos import QoSManager
from loraflexsim.scenarios.qos_cluster_bench import _create_simulator


def _snir_series(*, seed: int, fading_std_db: float | None) -> List[float]:
    """Exécute une courte simulation et retourne les SNIR moyens par paquet."""

    simulator = _create_simulator(
        8,
        3.0,
        seed,
        use_snir=True,
        channel_overrides={"snir_fading_std": fading_std_db} if fading_std_db is not None else None,
    )
    QoSManager().apply(simulator, "ADR-Pure", use_snir=True)
    simulator.run(max_time=25.0)

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


def test_snir_reproducibility_same_seed_and_change() -> None:
    series_first = _snir_series(seed=21, fading_std_db=None)
    series_second = _snir_series(seed=21, fading_std_db=None)
    _assert_same_sequence(series_second, series_first)

    changed_seed_series = _snir_series(seed=37, fading_std_db=None)
    assert changed_seed_series != series_first, "Les séries doivent diverger quand la graine change"


def test_snir_reproducibility_with_fading_variance() -> None:
    series_first = _snir_series(seed=11, fading_std_db=4.0)
    series_second = _snir_series(seed=11, fading_std_db=4.0)
    _assert_same_sequence(series_second, series_first)

    std_dev = statistics.pstdev(series_first)
    assert 0.5 <= std_dev <= 30.0, (
        "La variance SNIR avec fading attendu devrait rester dans une plage réaliste: "
        f"σ={std_dev:.2f} dB"
    )
