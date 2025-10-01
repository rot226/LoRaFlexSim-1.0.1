import pytest

from loraflexsim.launcher.adr_standard_1 import apply as apply_adr
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.lorawan import TX_POWER_INDEX_TO_DBM
from loraflexsim.launcher.simulator import Simulator


def _run(distance: float, initial_sf: int = 12, packets: int = 30):
    ch = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=packets,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[ch],
        seed=1,
    )
    apply_adr(sim)
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = 0.0
    node.y = 0.0
    gw.x = distance
    gw.y = 0.0
    node.sf = initial_sf
    node.initial_sf = initial_sf
    sim.run()
    return node


def test_adr_decreases_sf_with_good_link():
    node = _run(distance=1.0)
    assert node.sf == 7
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[4]


def test_adr_increases_sf_with_poor_link():
    node = _run(distance=10000.0, initial_sf=8)
    assert node.sf == 7
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[3]


def test_adr_recovers_after_rx2_rejection():
    from loraflexsim.launcher.lorawan import LinkADRReq, compute_rx2, SF_TO_DR

    ch = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=0,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[ch],
        seed=1,
    )
    apply_adr(sim)
    node = sim.nodes[0]
    gw = sim.gateways[0]
    scheduler = sim.network_server.scheduler
    node.security_enabled = False
    node.sf = 12
    node.last_uplink_end_time = 0.0

    rx2 = compute_rx2(node.last_uplink_end_time, node.rx_delay)
    scheduler._gateway_busy[gw.id] = rx2 + 1.0

    initial_pending = node.downlink_pending
    initial_fcnt = node.fcnt_down
    frames_before = node.frames_since_last_adr_command

    sim.network_server.send_downlink(
        node,
        adr_command=(7, node.tx_power, node.chmask, node.nb_trans),
        gateway=gw,
    )

    assert node.downlink_pending == initial_pending
    assert scheduler.next_time(node.id) is None
    assert node.fcnt_down == initial_fcnt
    assert node.frames_since_last_adr_command == frames_before

    scheduler._gateway_busy[gw.id] = 0.0
    sim.network_server.send_downlink(
        node,
        adr_command=(7, node.tx_power, node.chmask, node.nb_trans),
        gateway=gw,
    )

    scheduled_time = scheduler.next_time(node.id)
    assert scheduled_time is not None
    assert node.downlink_pending == initial_pending + 1

    entry = scheduler.pop_ready(node.id, scheduled_time)
    assert entry is not None
    req = LinkADRReq.from_bytes(entry.frame.payload[:5])
    assert req.datarate == SF_TO_DR[7]

    node.handle_downlink(entry.frame)
    assert node.sf == 7
    assert node.downlink_pending == initial_pending


@pytest.mark.parametrize("node_class", ["B", "C"])
def test_adr_scheduler_quantization(node_class: str):
    ch = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        node_class=node_class,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=40,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[ch],
        seed=1,
        tick_ns=1_000_000,
    )
    apply_adr(sim)
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = 0.0
    node.y = 0.0
    gw.x = 0.0
    gw.y = 0.0
    initial_sf = node.sf
    sim.run()
    assert node.sf < initial_sf
