import math

import pytest

from loraflexsim.launcher.simulator import Simulator


def test_simulator_rejects_non_positive_intervals():
    with pytest.raises(ValueError):
        Simulator(packet_interval=0.0)
    with pytest.raises(ValueError):
        Simulator(packet_interval=-1.0)
    with pytest.raises(ValueError):
        Simulator(packet_interval=10.0, first_packet_interval=0.0)
    with pytest.raises(ValueError):
        Simulator(packet_interval=10.0, first_packet_interval=math.inf)


def test_simulator_rejects_non_real_intervals():
    with pytest.raises(TypeError):
        Simulator(packet_interval=True)
    with pytest.raises(TypeError):
        Simulator(packet_interval=10.0, first_packet_interval="5")
