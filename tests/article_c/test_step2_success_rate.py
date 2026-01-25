from __future__ import annotations

from article_c.step2.simulate_step2 import run_simulation


def test_step2_success_rate_has_success_window() -> None:
    result = run_simulation(
        algorithm="adr",
        n_rounds=2,
        n_nodes=3,
        window_size=2,
        snir_mode="snir_on",
        seed=123,
        jitter_range_s=0.5,
        window_duration_s=8.0,
        traffic_coeff_enabled=False,
        window_delay_enabled=False,
    )
    success_rates = [
        float(row.get("success_rate", 0.0))
        for row in result.raw_rows
        if row.get("cluster") == "all"
    ]
    assert success_rates, "Aucune fenêtre disponible pour vérifier le success_rate."
    assert any(rate > 0.0 for rate in success_rates)
