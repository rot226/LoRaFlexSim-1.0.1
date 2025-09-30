import numpy_stub as np
import pytest

from traffic.exponential import sample_exp


def test_sample_exp_asserts():
    rng = np.random.Generator(np.random.MT19937(0))
    with pytest.raises(ValueError):
        sample_exp(-1.0, rng)
    with pytest.raises(ValueError):
        sample_exp(10, rng)  # not float
