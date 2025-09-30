from loraflexsim.launcher.simulator import Simulator
from scipy.stats import kstest


def uniform_cdf(x: float, low: float, high: float) -> float:
    """Simple CDF for a continuous uniform distribution on [low, high]."""
    if x < low:
        return 0.0
    if x > high:
        return 1.0
    return (x - low) / (high - low)


def test_node_positions_uniform_after_first_move():
    sim = Simulator(num_nodes=100, num_gateways=1, area_size=100.0, mobility=True, seed=123)
    sim.run(max_steps=1)

    coords = [(node.x, node.y) for node in sim.nodes]
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]

    _, p_x = kstest(xs, uniform_cdf, args=(0.0, sim.area_size))
    _, p_y = kstest(ys, uniform_cdf, args=(0.0, sim.area_size))

    assert p_x > 0.01 and p_y > 0.01
