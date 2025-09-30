import numpy_stub as np
import pytest
import math
from scipy import stats

from traffic.exponential import sample_interval


@pytest.mark.parametrize("seed", range(10))
def test_interval_distribution_mean(seed: int) -> None:
    """``sample_interval`` should preserve the configured mean.

    The test draws ``count`` samples from several RNG streams and checks that
    the empirical mean stays within 1% of ``mean_interval``. Using multiple
    seeds helps detect potential systematic bias in the implementation.
    """

    mean_interval = 10.0
    count = 1_000_000
    rng = np.random.Generator(np.random.MT19937(seed))
    total = 0.0
    for _ in range(count):
        total += sample_interval(mean_interval, rng)
    average = total / count
    assert abs(average - mean_interval) / mean_interval < 0.01


@pytest.mark.parametrize("seed", range(10))
def test_interval_distribution_cv(seed: int) -> None:
    """``sample_interval`` should have a coefficient of variation of 1."""

    mean_interval = 10.0
    count = 1_000_000
    rng = np.random.Generator(np.random.MT19937(seed))
    total = 0.0
    sumsq = 0.0
    for _ in range(count):
        x = sample_interval(mean_interval, rng)
        total += x
        sumsq += x * x
    mean = total / count
    variance = sumsq / count - mean * mean
    std = math.sqrt(variance)
    cv = std / mean
    assert abs(cv - 1.0) < 0.02


def _ks_p_value(samples, mean_interval):
    n = len(samples)
    samples = sorted(samples)
    d_plus = 0.0
    d_minus = 0.0
    for i, x in enumerate(samples, 1):
        cdf = 1.0 - math.exp(-x / mean_interval)
        d_plus = max(d_plus, i / n - cdf)
        d_minus = max(d_minus, cdf - (i - 1) / n)
    d = d_plus if d_plus > d_minus else d_minus
    p = 2 * math.exp(-2 * n * d * d)
    if p > 1.0:
        p = 1.0
    return p


@pytest.mark.parametrize("seed", range(3))
def test_interval_distribution_ks(seed: int) -> None:
    mean_interval = 10.0
    count = 10_000
    rng = np.random.Generator(np.random.MT19937(seed))
    samples = [sample_interval(mean_interval, rng) for _ in range(count)]
    p = _ks_p_value(samples, mean_interval)
    assert p > 0.05

@pytest.mark.parametrize("seed", range(10))
def test_interval_distribution_kstest(seed: int) -> None:
    mean_interval = 10.0
    count = 10_000
    rng = np.random.Generator(np.random.MT19937(seed))
    samples = [sample_interval(mean_interval, rng) for _ in range(count)]
    statistic, p = stats.kstest(samples, "expon", args=(0, mean_interval))
    assert p >= 0.05



def test_sample_interval_seed_reproducibility() -> None:
    rng1 = np.random.Generator(np.random.MT19937(123))
    rng2 = np.random.Generator(np.random.MT19937(123))
    seq1 = [sample_interval(5.0, rng1) for _ in range(1000)]
    seq2 = [sample_interval(5.0, rng2) for _ in range(1000)]
    assert seq1 == seq2
