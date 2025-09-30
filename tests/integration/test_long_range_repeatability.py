"""Statistiques de répétabilité pour le scénario longue portée."""

from __future__ import annotations

from statistics import mean

import pytest

try:
    pytest.importorskip("pandas")
except Exception:
    pytest.skip("pandas import failed", allow_module_level=True)

from loraflexsim.validation import (
    SCENARIOS,
    load_flora_reference,
    run_validation,
)


REPEAT_COUNT = 30


def _long_range_scenario():
    for scenario in SCENARIOS:
        if scenario.name == "long_range":
            return scenario
    raise AssertionError("Le scénario long_range est introuvable dans SCENARIOS")


def test_long_range_mean_pdr_matches_reference_with_margin() -> None:
    """La moyenne de 30 runs doit rester à ±0,01 du PDR de référence."""

    scenario = _long_range_scenario()
    reference = load_flora_reference(scenario.flora_reference)
    base_seed = scenario.sim_kwargs.get("seed")

    pdr_samples: list[float] = []
    for idx in range(REPEAT_COUNT):
        overrides = {}
        if base_seed is not None:
            overrides["seed"] = int(base_seed) + idx
        sim = scenario.build_simulator(**overrides)
        metrics = run_validation(sim, scenario.run_steps)
        pdr_samples.append(metrics["PDR"])

    average_pdr = mean(pdr_samples)
    delta = abs(average_pdr - reference["PDR"])
    assert (
        delta <= 0.01
    ), f"Moyenne PDR {average_pdr:.5f} trop éloignée de la référence {reference['PDR']:.5f} (Δ={delta:.5f})"
