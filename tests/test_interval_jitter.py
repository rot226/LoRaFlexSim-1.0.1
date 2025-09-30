import numpy_stub as np
from traffic.exponential import sample_interval
from loraflexsim.launcher.node import Node


def test_jitter_multiplier():
    rng = np.random.Generator(np.random.MT19937(0))
    node = Node(0, 0, 0, 7, 20)
    node.ensure_poisson_arrivals(10.0, 1.0, rng, variation=0.5, limit=1)
    delta = node.arrival_queue[0]

    rng2 = np.random.Generator(np.random.MT19937(0))
    base = sample_interval(1.0, rng2)
    factor = 1.0 + (2.0 * rng2.random() - 1.0) * 0.5
    if factor < 0.0:
        factor = 0.0
    expected = base * factor
    assert abs(delta - expected) < 1e-12
