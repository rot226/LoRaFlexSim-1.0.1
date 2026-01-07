from __future__ import annotations

from loraflexsim.launcher import Channel, Simulator


def test_baseline_loss_load_limits_pdr_der() -> None:
    channel = Channel(
        use_snir=True,
        baseline_loss_rate=0.03,
        baseline_collision_rate=0.01,
        residual_collision_load_scale=6.0,
    )
    channel.shadowing_std = 0.0
    simulator = Simulator(
        num_nodes=60,
        num_gateways=1,
        area_size=1000.0,
        packets_to_send=4,
        transmission_mode="Random",
        packet_interval=5.0,
        mobility=False,
        seed=321,
        channels=[channel],
    )
    simulator.run()
    metrics = simulator.get_metrics()
    total_sent = metrics.get("tx_attempted", 0) or 0
    delivered = metrics.get("delivered", 0) or 0
    der = delivered / total_sent if total_sent else 0.0
    assert metrics["PDR"] < 0.98
    assert der < 0.98
