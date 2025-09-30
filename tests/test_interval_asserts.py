import numpy_stub as np
import pytest

from traffic.exponential import sample_interval
from loraflexsim.launcher.node import Node


def test_sample_interval_asserts():
    rng = np.random.Generator(np.random.MT19937(0))
    with pytest.raises(ValueError):
        sample_interval(-1.0, rng)
    with pytest.raises(ValueError):
        sample_interval(10, rng)  # not float


def test_node_poisson_assert():
    rng = np.random.Generator(np.random.MT19937(0))
    node = Node(0, 0, 0, 7, 20)
    with pytest.raises(ValueError):
        node.ensure_poisson_arrivals(10.0, -5.0, rng)
    with pytest.raises(ValueError):
        node.ensure_poisson_arrivals(10.0, 5, rng)
    with pytest.raises(ValueError):
        node.ensure_poisson_arrivals(10.0, 5.0, rng, variation=-1.0)
