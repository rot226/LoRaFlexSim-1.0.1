from __future__ import annotations

import pytest

from article_c.step2.simulate_step2 import run_simulation


def test_step2_collision_regression_mini_case_n80() -> None:
    """Régression mini-cas: la charge provoque des collisions sans effondrer totalement le succès."""
    result = run_simulation(
        algorithm="adr",
        n_rounds=4,
        n_nodes=80,
        window_size=4,
        snir_mode="snir_on",
        seed=11,
        jitter_range_s=1.5,
        window_duration_s=16.0,
    )

    rows = [row for row in result.raw_rows if row.get("cluster") == "all"]
    assert rows, "Aucune ligne agrégée cluster=all disponible."

    successes = sum(int(row.get("success", 0)) for row in rows)
    failures = sum(int(row.get("failure", 0)) for row in rows)
    total = successes + failures
    assert total > 0, "Aucun paquet transmis: impossible de valider la régression."

    mean_success_rate = successes / total
    min_success_rate = 0.03
    if mean_success_rate <= min_success_rate:
        pytest.fail(
            "Effondrement détecté du succès moyen sur N=80: "
            f"mean_success_rate={mean_success_rate:.4f} <= seuil={min_success_rate:.4f}."
        )

    total_collisions = sum(int(row.get("total_collisions", 0)) for row in rows)
    assert total_collisions > 0, "Aucune collision observée alors que la charge est élevée."

    collision_norm_mean = sum(float(row.get("collision_norm", 0.0)) for row in rows) / len(rows)
    assert collision_norm_mean < 1.0, (
        "Les collisions dominent à 100% (collision_norm moyen == 1), "
        "ce qui indique une régression sévère."
    )

    total_capture_events = sum(int(row.get("capture_events", 0)) for row in rows)
    assert total_capture_events > 0, "Aucun capture effect observé sur le mini-cas N=80."
