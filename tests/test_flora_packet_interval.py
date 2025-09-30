from pathlib import Path

from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.config_loader import parse_flora_interval


def test_flora_packet_interval():
    ini = Path('flora-master/simulations/examples/n100-gw1.ini')
    mean_interval = parse_flora_interval(ini)
    assert mean_interval is not None

    sim = Simulator(
        flora_mode=True,
        config_file=str(ini),
        packets_to_send=300,
        mobility=False,
        seed=0,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert abs(metrics["avg_arrival_interval_s"] - mean_interval) / mean_interval < 0.01
