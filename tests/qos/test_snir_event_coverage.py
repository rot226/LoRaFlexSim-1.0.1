from __future__ import annotations

import math

from loraflexsim.scenarios.qos_cluster_bench import _create_simulator


def test_snir_event_coverage_for_collisions() -> None:
    simulator = _create_simulator(
        80,
        0.6,
        13,
        use_snir=True,
    )
    simulator.run(max_time=30.0)

    collision_events = [
        entry
        for entry in simulator.events_log
        if entry.get("result") == "CollisionLoss"
    ]
    assert collision_events, "Aucune collision détectée pour vérifier la couverture SNIR"

    for entry in collision_events:
        assert "snir_dB" in entry, "snir_dB manquant sur un événement de collision"
        snir_value = entry.get("snir_dB")
        assert snir_value is not None and math.isfinite(
            float(snir_value)
        ), "snir_dB doit être renseigné pour les collisions"
