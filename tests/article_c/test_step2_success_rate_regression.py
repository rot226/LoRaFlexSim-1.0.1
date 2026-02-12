from __future__ import annotations

from article_c.step2.simulate_step2 import run_simulation


def _overall_success_rate(n_nodes: int) -> float:
    result = run_simulation(
        algorithm="adr",
        n_rounds=8,
        n_nodes=n_nodes,
        window_size=4,
        snir_mode="snir_on",
        seed=17,
        jitter_range_s=2.0,
        window_duration_s=20.0,
    )
    rows = [row for row in result.raw_rows if row.get("cluster") == "all"]
    successes = sum(int(row.get("success", 0)) for row in rows)
    failures = sum(int(row.get("failure", 0)) for row in rows)
    total = successes + failures
    assert total > 0, "Aucun paquet transmis: impossible d'évaluer le taux de succès."
    return successes / total


def test_step2_success_rate_is_monotonic_with_network_size() -> None:
    node_sizes = (20, 40, 60, 80)
    rates = [_overall_success_rate(size) for size in node_sizes]
    for left, right in zip(rates, rates[1:]):
        assert left >= right, (
            "Le success_rate doit rester décroissant avec N. "
            f"Valeurs observées: {rates}."
        )


def test_step2_success_rate_stays_non_zero_at_n80() -> None:
    success_rate = _overall_success_rate(80)
    assert success_rate > 0.0
