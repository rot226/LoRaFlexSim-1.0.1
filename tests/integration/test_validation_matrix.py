"""Integration tests validating LoRaFlexSim against FLoRa baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

try:  # pragma: no cover - optional dependency
    import pandas as _pd  # noqa: F401
except Exception:  # pragma: no cover - skip if pandas unusable
    pytest.skip("pandas is required for validation comparisons", allow_module_level=True)

from loraflexsim.launcher import adr_ml, explora_at
from loraflexsim.validation import (
    SCENARIOS,
    compare_to_reference,
    load_flora_reference,
    run_validation,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_scenario_matches_flora_reference(scenario):
    """Each validation scenario stays within the tolerance bound."""

    assert Path(scenario.flora_config).exists(), f"Missing FLoRa config {scenario.flora_config}"
    assert Path(scenario.flora_reference).exists(), f"Missing reference file {scenario.flora_reference}"

    sim = scenario.build_simulator()
    metrics = run_validation(sim, scenario.run_steps)
    reference = load_flora_reference(scenario.flora_reference)
    deltas = compare_to_reference(metrics, reference, scenario.tolerances)

    assert deltas["PDR"] <= scenario.tolerances.pdr, (
        f"PDR delta {deltas['PDR']:.3f} exceeds tolerance {scenario.tolerances.pdr:.3f}"
    )
    assert deltas["collisions"] <= scenario.tolerances.collisions, (
        f"Collision delta {deltas['collisions']:.3f} exceeds tolerance {scenario.tolerances.collisions:.3f}"
    )
    assert deltas["snr"] <= scenario.tolerances.snr, (
        f"SNR delta {deltas['snr']:.3f} exceeds tolerance {scenario.tolerances.snr:.3f}"
    )


def test_multi_gateway_adr_alignment_matches_flora():
    """Average SNIR from each gateway matches FLoRa for the multi-GW scenario."""

    scenario = next(
        sc for sc in SCENARIOS if sc.name == "multi_gw_multichannel_server_adr"
    )
    sim = scenario.build_simulator()
    run_validation(sim, scenario.run_steps)
    reference = load_flora_reference(scenario.flora_reference)

    ns = sim.network_server
    snir_samples: list[float] = []
    for per_event in ns.gateway_snr_samples.values():
        snir_samples.extend(per_event.values())

    assert snir_samples, "No SNIR samples recorded for gateways"
    avg_snir = sum(snir_samples) / len(snir_samples)
    assert avg_snir == pytest.approx(
        reference["snr"], abs=scenario.tolerances.snr
    )


def test_validation_matrix_covers_specialised_modules():
    """Each advanced module is represented by at least one scenario."""

    def _has(predicate):
        return any(predicate(sc) for sc in SCENARIOS)

    assert _has(
        lambda sc: sc.sim_kwargs.get("duty_cycle") not in (None, 0)
    ), "Duty-cycle scenario missing"
    assert _has(
        lambda sc: sc.sim_kwargs.get("channel_distribution") == "random"
    ), "Dynamic multichannel scenario missing"
    assert _has(
        lambda sc: sc.sim_kwargs.get("node_class") == "B"
        and (
            sc.sim_kwargs.get("mobility")
            or sc.sim_kwargs.get("mobility_model") is not None
        )
    ), "Class B mobility scenario missing"
    assert _has(
        lambda sc: sc.sim_kwargs.get("node_class") == "C"
        and (
            sc.sim_kwargs.get("mobility")
            or sc.sim_kwargs.get("mobility_model") is not None
        )
    ), "Class C mobility scenario missing"
    assert _has(
        lambda sc: any(hook is explora_at.apply for hook in getattr(sc, "setup", ()))
    ), "EXPLoRa scenario missing"
    assert _has(
        lambda sc: any(hook is adr_ml.apply for hook in getattr(sc, "setup", ()))
    ), "ADR-ML scenario missing"
