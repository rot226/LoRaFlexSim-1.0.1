from loraflexsim.launcher.simulator import Simulator
from scipy.stats import kstest


def uniform_cdf(x: float, low: float, high: float) -> float:
    """Simple CDF for a continuous uniform distribution on [low, high]."""
    if x < low:
        return 0.0
    if x > high:
        return 1.0
    return (x - low) / (high - low)


def test_node_position_determinism():
    params = dict(num_nodes=5, num_gateways=1, area_size=100.0, mobility=False, seed=123)

    sim1 = Simulator(**params)
    sim2 = Simulator(**params)

    coords1 = [(node.x, node.y) for node in sim1.nodes]
    coords2 = [(node.x, node.y) for node in sim2.nodes]

    assert coords1 == coords2

    xs = [x for x, _ in coords1]
    ys = [y for _, y in coords1]

    _, p_x = kstest(xs, uniform_cdf, args=(0.0, params["area_size"]))
    _, p_y = kstest(ys, uniform_cdf, args=(0.0, params["area_size"]))

    assert p_x > 0.01 and p_y > 0.01
