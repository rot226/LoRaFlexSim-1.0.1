import sys
import random
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.compare_flora import load_flora_rx_stats  # noqa: E402


def _make_colliding_sim() -> Simulator:
    ch = Channel(shadowing_std=0, fast_fading_std=0, fine_fading_std=1.0,
                 variable_noise_std=0.5)
    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    gw = sim.gateways[0]
    for node in sim.nodes:
        node.x = gw.x
        node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    for node in sim.nodes:
        sim.schedule_event(node, 0.0)
    return sim


def test_rssi_snr_and_collisions_against_flora():
    random.seed(0)
    sim = _make_colliding_sim()
    while sim.step():
        pass
    rssi = sim.nodes[0].last_rssi
    snr = sim.nodes[0].last_snr
    flora_csv = Path(__file__).parent / "data" / "flora_rx_stats.csv"
    flora = load_flora_rx_stats(flora_csv)
    assert rssi == pytest.approx(flora["rssi"], abs=1e-3)
    assert snr == pytest.approx(flora["snr"], abs=1e-3)
    assert sim.packets_lost_collision == flora["collisions"]

