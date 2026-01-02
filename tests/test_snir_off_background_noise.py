from __future__ import annotations

from loraflexsim.launcher import Channel, Simulator


def test_snir_off_pdr_der_not_perfect() -> None:
    channel = Channel(use_snir=False)
    channel.shadowing_std = 0.0
    simulator = Simulator(
        num_nodes=80,
        num_gateways=1,
        area_size=1000.0,
        packets_to_send=5,
        transmission_mode="Random",
        packet_interval=4.0,
        mobility=False,
        seed=123,
        channels=[channel],
    )
    simulator.run()
    metrics = simulator.get_metrics()
    total_sent = metrics.get("tx_attempted", 0) or 0
    delivered = metrics.get("delivered", 0) or 0
    der = delivered / total_sent if total_sent else 0.0
    assert metrics["PDR"] <= 0.98
    assert der <= 0.98
