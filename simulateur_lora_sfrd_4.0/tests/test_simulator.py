import math
import sys
from pathlib import Path
import heapq

import pytest
import random

# Allow importing the VERSION_4 package from the repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator, EventType, Event  # noqa: E402
from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.server import NetworkServer  # noqa: E402
from VERSION_4.launcher.lorawan import (  # noqa: E402
    LinkADRAns,
    LinkCheckReq,
    LinkCheckAns,
    DeviceTimeReq,
)


def test_channel_compute_rssi_and_airtime():
    ch = Channel(shadowing_std=0)
    rssi, snr = ch.compute_rssi(14.0, 100.0)
    expected_rssi = 14.0 - ch.path_loss(100.0) - ch.cable_loss_dB
    expected_snr = expected_rssi - ch.noise_floor_dBm()
    assert rssi == pytest.approx(expected_rssi, rel=1e-6)
    assert snr == pytest.approx(expected_snr, rel=1e-6)

    at = ch.airtime(sf=7, payload_size=20)
    rs = ch.bandwidth / (2 ** 7)
    ts = 1.0 / rs
    de = 0
    cr_denom = ch.coding_rate + 4
    numerator = 8 * 20 - 4 * 7 + 28 + 16 - 20 * 0
    denominator = 4 * (7 - 2 * de)
    n_payload = max(math.ceil(numerator / denominator), 0) * cr_denom + 8
    expected_at = (ch.preamble_symbols + 4.25) * ts + n_payload * ts
    assert at == pytest.approx(expected_at, rel=1e-6)


def _make_sim(num_nodes: int, same_start: bool, min_interference_time: float = 0.0) -> Simulator:
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=num_nodes,
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
        min_interference_time=min_interference_time,
    )
    gw = sim.gateways[0]
    for n in sim.nodes:
        n.x = gw.x
        n.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    if same_start:
        for node in sim.nodes:
            sim.schedule_event(node, 0.0)
    else:
        for idx, node in enumerate(sim.nodes):
            sim.schedule_event(node, idx * 1.0)
    return sim


def test_simulator_step_success():
    sim = _make_sim(num_nodes=1, same_start=False)
    while sim.step():
        pass
    node = sim.nodes[0]
    assert sim.packets_delivered == 1
    assert sim.network_server.packets_received == 1
    assert node.packets_success == 1


def test_simulator_step_collision():
    sim = _make_sim(num_nodes=2, same_start=True)
    while sim.step():
        pass
    assert sim.packets_delivered == 0
    assert sim.packets_lost_collision == 2
    for node in sim.nodes:
        assert node.packets_collision == 1


def test_metrics_retransmissions():
    sim = _make_sim(num_nodes=1, same_start=False)
    sim.nodes[0].nb_trans = 2
    while sim.step():
        pass
    metrics = sim.get_metrics()
    assert metrics["retransmissions"] == 1
    assert sim.packets_sent == 2


def test_metrics_pdr_by_class():
    sim = _make_sim(num_nodes=2, same_start=False)
    sim.nodes[0].class_type = "A"
    sim.nodes[1].class_type = "C"
    while sim.step():
        pass
    metrics = sim.get_metrics()
    assert "pdr_by_class" in metrics
    assert set(metrics["pdr_by_class"].keys()) == {"A", "C"}
    assert metrics["pdr_by_class"]["A"] == pytest.approx(1.0)
    assert metrics["pdr_by_class"]["C"] == pytest.approx(1.0)


def test_lorawan_frame_handling():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    up = node.prepare_uplink(b"ping", confirmed=True)
    assert up.confirmed
    assert node.fcnt_up == 1
    assert node.awaiting_ack is True

    gw = Gateway(1, 0.0, 0.0)
    server = NetworkServer()
    server.gateways = [gw]
    server.nodes = [node]
    server.send_downlink(node, b"", confirmed=True, adr_command=(9, 4.0), request_ack=True)

    down = gw.pop_downlink(node.id)
    assert down is not None
    node.handle_downlink(down)
    assert node.sf == 9
    assert node.tx_power == 4.0
    assert node.pending_mac_cmd == LinkADRAns().to_bytes()
    assert node.awaiting_ack is False
    assert node.need_downlink_ack

    up2 = node.prepare_uplink(b"data")
    assert up2.fctrl & 0x20
    assert up2.payload.startswith(LinkADRAns().to_bytes())
    assert node.pending_mac_cmd is None
    assert not node.need_downlink_ack


def test_downlink_ack_bit_and_mac_commands():
    node = Node(2, 0.0, 0.0, 7, 14.0, channel=Channel())
    gw = Gateway(1, 0.0, 0.0)
    server = NetworkServer()
    server.gateways = [gw]
    server.nodes = [node]

    node.prepare_uplink(b"foo", confirmed=True)
    assert node.awaiting_ack

    server.send_downlink(
        node,
        LinkCheckReq().to_bytes(),
        confirmed=True,
        request_ack=True,
    )
    frame = gw.pop_downlink(node.id)
    node.handle_downlink(frame)
    assert not node.awaiting_ack
    assert node.need_downlink_ack
    assert node.pending_mac_cmd == LinkCheckAns(margin=255, gw_cnt=1).to_bytes()

    up = node.prepare_uplink(b"hello")
    assert up.fctrl & 0x20
    assert up.payload.startswith(LinkCheckAns(margin=255, gw_cnt=1).to_bytes())
    assert not node.need_downlink_ack

    # DeviceTimeReq
    server.send_downlink(node, DeviceTimeReq().to_bytes())
    frame2 = gw.pop_downlink(node.id)
    node.handle_downlink(frame2)
    assert node.pending_mac_cmd is not None


def test_sim_run_and_step_equivalence():
    random.seed(12345)
    sim_run = _make_sim(num_nodes=3, same_start=False)
    sim_run.run()
    metrics_run = sim_run.get_metrics()

    random.seed(12345)
    sim_step = _make_sim(num_nodes=3, same_start=False)
    while sim_step.step():
        pass
    metrics_step = sim_step.get_metrics()

    keys = [
        "PDR",
        "collisions",
        "energy_J",
        "avg_delay_s",
        "retransmissions",
        "throughput_bps",
    ]
    for key in keys:
        assert metrics_run[key] == pytest.approx(metrics_step[key])


def test_get_events_dataframe_has_all_columns():
    pytest.importorskip("pandas")
    sim = _make_sim(num_nodes=1, same_start=False)
    sim.run()
    df = sim.get_events_dataframe()
    assert not df.empty
    expected_columns = [
        "event_id",
        "node_id",
        "initial_x",
        "initial_y",
        "final_x",
        "final_y",
        "initial_sf",
        "final_sf",
        "initial_tx_power",
        "final_tx_power",
        "packets_sent",
        "packets_success",
        "packets_collision",
        "energy_consumed_J_node",
        "battery_capacity_J",
        "battery_remaining_J",
        "downlink_pending",
        "acks_received",
        "start_time",
        "end_time",
        "energy_J",
        "rssi_dBm",
        "snr_dB",
        "result",
        "gateway_id",
    ]
    for col in expected_columns:
        assert col in df.columns


def test_simulator_seed_reproducibility():
    ch = Channel(shadowing_std=0)
    kwargs = dict(
        num_nodes=3,
        num_gateways=2,
        area_size=50.0,
        transmission_mode="Random",
        packet_interval=10.0,
        packets_to_send=0,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    sim1 = Simulator(**kwargs, seed=42)
    sim2 = Simulator(**kwargs, seed=42)
    pos1 = [(n.x, n.y) for n in sim1.nodes]
    pos2 = [(n.x, n.y) for n in sim2.nodes]
    gw1 = [(g.x, g.y) for g in sim1.gateways]
    gw2 = [(g.x, g.y) for g in sim2.gateways]
    assert pos1 == pos2
    assert gw1 == gw2


def _downlink_exchange(node: Node, gw: Gateway, ch: Channel, deliver: bool):
    """Helper to simulate a downlink reception or loss."""
    server = NetworkServer()
    server.gateways = [gw]
    server.nodes = [node]
    server.channel = ch
    server.send_downlink(node, b"data")
    assert node.downlink_pending == 1

    frame = gw.pop_downlink(node.id)
    assert frame is not None
    distance = node.distance_to(gw)
    rssi, snr = ch.compute_rssi(node.tx_power, distance)
    snr_threshold = ch.sensitivity_dBm.get(node.sf, -float("inf")) - ch.noise_floor_dBm()
    if deliver and rssi >= ch.detection_threshold_dBm and snr >= snr_threshold:
        node.handle_downlink(frame)
    else:
        node.downlink_pending = max(0, node.downlink_pending - 1)


def test_downlink_pending_decrement_success():
    ch = Channel(shadowing_std=0)
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=ch)
    gw = Gateway(1, 0.0, 0.0)
    _downlink_exchange(node, gw, ch, deliver=True)
    assert node.downlink_pending == 0


def test_downlink_pending_decrement_failure():
    ch = Channel(shadowing_std=0)
    node = Node(2, 0.0, 0.0, 7, 14.0, channel=ch)
    gw = Gateway(1, 15000.0, 0.0)
    _downlink_exchange(node, gw, ch, deliver=False)
    assert node.downlink_pending == 0


def test_detection_threshold_blocks_packet():
    ch = Channel(shadowing_std=0, detection_threshold_dBm=-110)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=5000.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = 3000.0
    node.y = 0.0
    gw.x = 0.0
    gw.y = 0.0
    sim.event_queue.clear()
    sim.event_id_counter = 0
    sim.schedule_event(node, 0.0)
    while sim.step():
        pass
    assert sim.packets_delivered == 0
    assert sim.packets_lost_no_signal == 1


def test_min_interference_time_avoids_collision():
    sim = _make_sim(num_nodes=2, same_start=True, min_interference_time=1.0)
    while sim.step():
        pass
    assert sim.packets_delivered == 2
    assert sim.packets_lost_collision == 0
    for node in sim.nodes:
        assert node.packets_success == 1


def test_duty_cycle_enforces_delay():
    """Next transmission should be postponed when duty cycle is enabled."""
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=0.1,
        packets_to_send=2,
        mobility=False,
        duty_cycle=0.1,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = gw.x
    node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    sim.schedule_event(node, 0.0)

    # Process until the next transmission for the same node is scheduled
    first_tx_id = 0
    while sim.step():
        if sim.event_queue and sim.event_queue[0].type == EventType.TX_START and sim.event_queue[0].id != first_tx_id:
            break

    next_time = sim.event_queue[0].time
    # With duty cycle 10%, airtime ~0.0566s => next allowed time ~0.566s
    assert next_time == pytest.approx(0.566, rel=0.05)


def test_prepare_uplink_join_request():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), activated=False)
    frame = node.prepare_uplink(b"payload")
    from VERSION_4.launcher.lorawan import JoinRequest

    assert isinstance(frame, JoinRequest)
    assert node.devnonce == 1


def test_network_server_activation():
    ch = Channel(shadowing_std=0)
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=ch, activated=False)
    gw = Gateway(1, 0.0, 0.0)
    server = NetworkServer()
    server.gateways = [gw]
    server.nodes = [node]
    server.channel = ch

    server.receive(0, node.id, gw.id)
    frame = gw.pop_downlink(node.id)
    from VERSION_4.launcher.lorawan import JoinAccept

    assert isinstance(frame, JoinAccept)
    node.handle_downlink(frame)
    assert node.activated


def test_simulator_join_accept_downlink():
    sim = _make_sim(num_nodes=1, same_start=False)
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.activated = False
    node.devaddr = None
    node.x = gw.x
    node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    sim.schedule_event(node, 0.0)
    while sim.step():
        pass
    assert node.activated


def test_beacon_schedules_ping_slot():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    sim.step()  # process beacon
    assert any(evt.type == EventType.PING_SLOT for evt in sim.event_queue)


def test_beacon_schedules_multiple_ping_slots():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    sim.step()  # process beacon
    ping_slots = [evt for evt in sim.event_queue if evt.type == EventType.PING_SLOT]
    assert len(ping_slots) > 1


def test_beacon_loss_does_not_stop_ping_slot():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    node.beacon_loss_prob = 1.0
    sim.step()  # process beacon
    assert any(evt.type == EventType.PING_SLOT for evt in sim.event_queue)


def test_class_c_rx_interval_setting():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
        class_c_rx_interval=2.0,
    )
    node = sim.nodes[0]
    node.class_type = "C"
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    sim.step()
    rx_events = [e for e in sim.event_queue if e.type == EventType.RX_WINDOW]
    assert rx_events and rx_events[0].time == pytest.approx(2.0)


def test_class_c_continuous_rx_energy():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "C"
    node.state = "rx"
    node.last_state_time = 0.0
    sim.event_queue.clear()
    sim.event_id_counter = 0
    eid = sim.event_id_counter
    sim.event_id_counter += 1
    heapq.heappush(sim.event_queue, Event(10.0, EventType.RX_WINDOW, eid, node.id))
    sim.step()
    expected = node.profile.rx_current_a * node.profile.voltage_v * 10.0
    assert node.energy_rx == pytest.approx(expected)
    assert node.state == "rx"


def test_ping_slot_periodicity_respected():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    node.ping_slot_periodicity = 1
    sim.step()  # process beacon
    ping_slots = [e for e in sim.event_queue if e.type == EventType.PING_SLOT]
    assert ping_slots[0].time == pytest.approx(sim.ping_slot_offset)
    assert ping_slots[1].time == pytest.approx(sim.ping_slot_offset + 2 * sim.ping_slot_interval)


def test_downlink_delivered_in_ping_slot():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    gw = sim.gateways[0]
    node.x = gw.x
    node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    heapq.heappush(sim.event_queue, Event(0.0, EventType.BEACON, 1, 0))
    sim.network_server.send_downlink(node, b"data", at_time=sim.ping_slot_offset)
    while sim.step():
        if sim.current_time > sim.ping_slot_offset + 0.1:
            break
    assert node.fcnt_down == 1
    assert node.downlink_pending == 0


def test_quasi_continuous_ping_slots():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
    node = sim.nodes[0]
    node.class_type = "B"
    node.ping_slot_periodicity = 0
    sim.step()  # process beacon
    ping_slots = [e for e in sim.event_queue if e.type == EventType.PING_SLOT]
    assert len(ping_slots) > 5
    assert ping_slots[1].time - ping_slots[0].time == pytest.approx(sim.ping_slot_interval)


def test_clock_accuracy_assigns_drift():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=3,
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
        clock_accuracy=0.001,
    )
    assert any(n.beacon_drift != 0.0 for n in sim.nodes)


def test_beacon_loss_prob_parameter():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
        beacon_loss_prob=0.5,
    )
    assert sim.nodes[0].beacon_loss_prob == pytest.approx(0.5)


def test_custom_ping_slot_timing():
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
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
        ping_slot_interval=2.0,
        ping_slot_offset=3.0,
    )
    node = sim.nodes[0]
    node.class_type = "B"
    node.ping_slot_periodicity = 1
    sim.step()
    ping_slots = sorted(
        [e for e in sim.event_queue if e.type == EventType.PING_SLOT],
        key=lambda e: e.time,
    )
    assert ping_slots[0].time == pytest.approx(sim.ping_slot_offset)
    assert ping_slots[1].time == pytest.approx(
        sim.ping_slot_offset + 2 * sim.ping_slot_interval
    )
