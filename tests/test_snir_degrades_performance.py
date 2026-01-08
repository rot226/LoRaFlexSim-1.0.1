from __future__ import annotations

from loraflexsim.scenarios.qos_cluster_bench import _create_simulator


MIN_SNR_GAP = 0.02


def _run_metrics(*, seed: int, use_snir: bool) -> tuple[float, float]:
    channel_overrides = {
        "baseline_loss_rate": 0.03,
        "baseline_collision_rate": 0.02,
        "residual_collision_prob": 0.08,
        "snir_off_noise_prob": 0.0,
        "snir_fading_std": 3.0,
        "marginal_snir_margin_db": 1.0,
        "marginal_snir_drop_prob": 0.5,
    }
    simulator = _create_simulator(
        80,
        5.0,
        seed,
        use_snir=use_snir,
        channel_overrides=channel_overrides,
    )
    simulator.run(max_time=200.0)
    metrics = simulator.get_metrics()
    total_sent = metrics.get("tx_attempted", 0) or 0
    delivered = metrics.get("delivered", 0) or 0
    der = delivered / total_sent if total_sent else 0.0
    return float(metrics["PDR"]), float(der)


def test_snir_on_degrades_pdr_der() -> None:
    pdr_on, der_on = _run_metrics(seed=77, use_snir=True)
    pdr_off, der_off = _run_metrics(seed=77, use_snir=False)

    assert pdr_on <= pdr_off - MIN_SNR_GAP, (
        f"PDR SNIR on ({pdr_on:.3f}) devrait être inférieure à SNIR off ({pdr_off:.3f})."
    )
    assert der_on <= der_off - MIN_SNR_GAP, (
        f"DER SNIR on ({der_on:.3f}) devrait être inférieure à SNIR off ({der_off:.3f})."
    )
    assert pdr_off < 1.0, "La PDR SNIR off doit rester < 1.0 même sans dégradation SNIR."
