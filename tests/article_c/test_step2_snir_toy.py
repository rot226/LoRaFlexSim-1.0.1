from __future__ import annotations

from article_c.step2.simulate_step2 import run_simulation


def _overall_success_rate(*, snir_threshold_db: float) -> float:
    result = run_simulation(
        algorithm="adr",
        n_rounds=8,
        n_nodes=40,
        window_size=4,
        snir_mode="snir_on",
        snir_threshold_db=snir_threshold_db,
        snir_threshold_min_db=-20.0,
        snir_threshold_max_db=20.0,
        seed=7,
        jitter_range_s=0.0,
        window_duration_s=8.0,
        traffic_coeff_enabled=False,
        window_delay_enabled=False,
    )
    rows = [row for row in result.raw_rows if row.get("cluster") == "all"]
    successes = sum(int(row.get("success", 0)) for row in rows)
    failures = sum(int(row.get("failure", 0)) for row in rows)
    total = successes + failures
    assert total > 0, "Aucun paquet transmis: impossible d'évaluer le taux de succès."
    return successes / total


def test_step2_toy_snir_eleve_has_positive_success_rate() -> None:
    success_rate = _overall_success_rate(snir_threshold_db=-20.0)
    assert success_rate > 0.0


def test_step2_toy_snir_faible_success_is_close_to_zero() -> None:
    tolerance = 0.05
    success_rate = _overall_success_rate(snir_threshold_db=20.0)
    assert success_rate <= tolerance
