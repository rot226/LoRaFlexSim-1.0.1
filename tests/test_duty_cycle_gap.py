import pytest

from loraflexsim.launcher.lorawan import DutyCycleReq, LoRaWANFrame
from loraflexsim.launcher.simulator import EventType, Simulator


def test_duty_cycle_command_updates_manager():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=0,
        mobility=False,
        duty_cycle=1.0,
        seed=0,
    )
    node = sim.nodes[0]
    while not node.in_transmission:
        assert sim.step()
    while node.in_transmission:
        assert sim.step()
    frame = LoRaWANFrame(
        mhdr=0x60,
        fctrl=0,
        fcnt=0,
        payload=DutyCycleReq(max_duty_cycle=8).to_bytes(),
        confirmed=False,
    )
    node.handle_downlink(frame)
    assert sim.duty_cycle_manager is not None
    expected_duty_cycle = 1 / (2 ** 8)
    assert sim.duty_cycle_manager.duty_cycle == pytest.approx(expected_duty_cycle)
    expected_deadline = max(
        sim.current_time, node.last_tx_time + node.last_airtime / expected_duty_cycle
    )
    assert sim.duty_cycle_manager.next_tx_time[node.id] == pytest.approx(expected_deadline)

    tx_events = [
        evt
        for evt in sim.event_queue
        if evt.node_id == node.id and evt.type == EventType.TX_START
    ]
    assert tx_events
    next_tx_time = min(sim._ticks_to_seconds(evt.time) for evt in tx_events)
    assert next_tx_time == pytest.approx(sim._quantize(expected_deadline))
    assert node.interval_log
    assert node.interval_log[-1]["reason"] == "duty_cycle"
    assert node.interval_log[-1]["tx_time"] == pytest.approx(sim._quantize(expected_deadline))
