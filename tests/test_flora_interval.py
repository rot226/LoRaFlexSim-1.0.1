from pathlib import Path

from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.config_loader import (
    parse_flora_interval,
    parse_flora_first_interval,
    load_config,
)


def test_flora_ini_interval_parsing():
    ini_path = Path('flora-master/simulations/examples/n100-gw1.ini')
    next_mean = parse_flora_interval(ini_path)
    first_mean = parse_flora_first_interval(ini_path)
    assert next_mean is not None
    assert first_mean == next_mean
    nodes, gws, next_from_load, first_from_load = load_config(ini_path)
    assert next_from_load == next_mean
    assert first_from_load == next_mean
    sim = Simulator(flora_mode=True, config_file=str(ini_path))
    assert sim.packet_interval == next_mean
    assert sim.first_packet_interval == sim.packet_interval
