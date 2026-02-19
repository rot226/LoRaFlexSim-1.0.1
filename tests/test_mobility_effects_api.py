import random

import numpy as np
import pytest

from loraflexsim.launcher.mobility_effects import mobility_penalty, stochastic_variation


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
