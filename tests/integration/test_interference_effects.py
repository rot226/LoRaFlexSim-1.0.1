"""Integration check for interference impact on collisions and DER."""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.snir_stage1_compare.scenarios.run_compare_stage1 import (
    SimulationTask,
    _run_single,
)


@pytest.mark.integration
def test_interference_impacts_collisions_and_der() -> None:
    base_task = SimulationTask(
        algorithm="snir",
        phy_profile="flora_full",
        num_nodes=400,
        packet_interval=10.0,
        seed=7,
        rep=1,
    )

    without_interference = _run_single(base_task, baseline_der_bias=False)
    with_interference = _run_single(
        replace(base_task, algorithm="snir_interference"), baseline_der_bias=False
    )

    assert without_interference.interference_enabled is False
    assert with_interference.interference_enabled is True

    collisions_delta = with_interference.collisions - without_interference.collisions
    der_delta = without_interference.der - with_interference.der

    assert collisions_delta >= 1000
    assert der_delta >= 0.0001
