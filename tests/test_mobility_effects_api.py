import random

import numpy as np
import pytest

from loraflexsim.launcher.mobility_effects import (
    generate_fig1_pdr_vs_speed,
    generate_fig4_der_vs_speed,
    mobility_penalty,
    stochastic_variation,
)


@pytest.mark.parametrize("model", ["sm", "smooth", "rwp", "random_waypoint"])
def test_penalty_is_monotonic_with_speed(model: str) -> None:
    speeds = [0.0, 1.0, 3.0, 7.0, 12.0]
    penalties = [mobility_penalty(model, speed) for speed in speeds]
    assert penalties == sorted(penalties)


def test_rwp_always_more_degrading_than_sm() -> None:
    for speed in np.linspace(0.0, 20.0, num=51):
        assert mobility_penalty("rwp", float(speed)) > mobility_penalty("sm", float(speed))


def test_stochastic_variation_stays_small_and_bounded() -> None:
    rng = np.random.default_rng(123)
    scale = 1.0
    values = [stochastic_variation("rwp", rng, scale) for _ in range(2000)]
    sigma = 0.018 * scale
    assert max(abs(v) for v in values) <= 3.0 * sigma + 1e-12
    assert abs(float(np.mean(values))) < 0.003


def test_stochastic_variation_supports_python_random() -> None:
    rng = random.Random(7)
    value = stochastic_variation("sm", rng, 0.5)
    assert isinstance(value, float)


def test_generate_fig1_pdr_vs_speed_respects_constraints() -> None:
    rng = np.random.default_rng(11)
    df = generate_fig1_pdr_vs_speed({"speeds": [0, 2, 4, 8, 12]}, rng)

    assert list(df.columns) == ["speed", "pdr_sm", "pdr_rwp"]
    assert np.all(np.diff(df["speed"]) > 0)
    assert np.all(np.diff(df["pdr_sm"]) <= 1e-12)
    assert np.all(np.diff(df["pdr_rwp"]) <= 1e-12)
    assert np.all(df["pdr_rwp"] < df["pdr_sm"])
    assert np.all((df[["pdr_sm", "pdr_rwp"]] >= 0.0).to_numpy())
    assert np.all((df[["pdr_sm", "pdr_rwp"]] <= 1.0).to_numpy())


def test_generate_fig4_der_vs_speed_respects_constraints() -> None:
    rng = random.Random(22)
    df = generate_fig4_der_vs_speed({"min_speed": 0, "max_speed": 10, "num_points": 6}, rng)

    assert list(df.columns) == ["speed", "der_sm", "der_rwp"]
    assert np.all(np.diff(df["speed"]) > 0)
    assert np.all(np.diff(df["der_sm"]) <= 1e-12)
    assert np.all(np.diff(df["der_rwp"]) <= 1e-12)
    assert np.all(df["der_rwp"] < df["der_sm"])
    assert np.all((df[["der_sm", "der_rwp"]] >= 0.0).to_numpy())
    assert np.all((df[["der_sm", "der_rwp"]] <= 1.0).to_numpy())


@pytest.mark.parametrize(
    "func,args",
    [
        (mobility_penalty, ("sm", -1.0)),
        (stochastic_variation, ("sm", np.random.default_rng(0), -0.1)),
        (mobility_penalty, ("unknown", 1.0)),
    ],
)
def test_invalid_inputs_raise_value_error(func, args) -> None:
    with pytest.raises(ValueError):
        func(*args)
