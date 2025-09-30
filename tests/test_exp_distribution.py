import numpy_stub as np
import pytest
import math

from traffic.exponential import sample_exp


@pytest.mark.parametrize("seed", range(10))
def test_exp_distribution_mean(seed: int) -> None:
    """``sample_exp`` should preserve the configured mean."""

    mu_send = 10.0
    count = 1_000_000
    rng = np.random.Generator(np.random.MT19937(seed))
    total = 0.0
    for _ in range(count):
        total += sample_exp(mu_send, rng)
    average = total / count
    assert abs(average - mu_send) / mu_send < 0.01


@pytest.mark.parametrize("seed", range(10))
def test_exp_distribution_cv(seed: int) -> None:
    """``sample_exp`` should have a coefficient of variation of 1."""

    mu_send = 10.0
    count = 1_000_000
    rng = np.random.Generator(np.random.MT19937(seed))
    total = 0.0
    sumsq = 0.0
    for _ in range(count):
        x = sample_exp(mu_send, rng)
        total += x
        sumsq += x * x
    mean = total / count
    variance = sumsq / count - mean * mean
    std = math.sqrt(variance)
    cv = std / mean
    assert abs(cv - 1.0) < 0.02
