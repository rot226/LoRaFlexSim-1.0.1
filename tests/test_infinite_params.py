import numpy as np
import pytest

from traffic.exponential import sample_interval, sample_exp
from loraflexsim.run import simulate
from traffic.rng_manager import RngManager


def _rng():
    return np.random.Generator(np.random.MT19937(0))


def test_sample_interval_rejects_infinite_mean():
    with pytest.raises(ValueError):
        sample_interval(float('inf'), _rng())


def test_sample_exp_rejects_infinite_mean():
    with pytest.raises(ValueError):
        sample_exp(float('inf'), _rng())


def test_simulate_rejects_infinite_interval():
    rng = RngManager(0)
    with pytest.raises(ValueError):
        simulate(nodes=1, gateways=1, mode="Random", interval=float('inf'), steps=10, rng_manager=rng)


def test_simulate_rejects_infinite_first_interval():
    rng = RngManager(0)
    with pytest.raises(ValueError):
        simulate(nodes=1, gateways=1, mode="Periodic", interval=1.0, steps=10, first_interval=float('inf'), rng_manager=rng)
