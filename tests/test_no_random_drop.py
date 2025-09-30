"""Integration tests ensuring a noiseless channel delivers every packet."""

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.simulator import EventType, Simulator


def make_clean_channel() -> Channel:
    """Return a channel without any random impairments."""

    return Channel(
        shadowing_std=0.0,
        fast_fading_std=0.0,
        time_variation_std=0.0,
        variable_noise_std=0.0,
        noise_floor_std=0.0,
        multipath_taps=1,
        impulsive_noise_prob=0.0,
        impulsive_noise_dB=0.0,
        phase_noise_std_dB=0.0,
        clock_jitter_std_s=0.0,
        pa_ramp_up_s=0.0,
        pa_ramp_down_s=0.0,
        fine_fading_std=0.0,
    )


def test_clean_channel_uplink_delivery():
    channel = make_clean_channel()
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=5,
        mobility=False,
        channels=[channel],
        fixed_sf=7,
        fixed_tx_power=14,
        seed=123,
    )

    sim.run()
    node = sim.nodes[0]

    assert node.tx_attempted == 5
    assert node.rx_delivered == node.tx_attempted
    assert sim.packets_lost_collision == 0
    assert sim.packets_lost_no_signal == 0


def test_clean_channel_rx_window_downlink_delivery():
    channel = make_clean_channel()
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        channels=[channel],
        fixed_sf=7,
        fixed_tx_power=14,
        seed=321,
    )

    node = sim.nodes[0]
    downlink_scheduled = False

    while sim.event_queue:
        next_event = sim.event_queue[0]
        if not downlink_scheduled and next_event.type == EventType.RX_WINDOW:
            sim.network_server.send_downlink(node, b"ping")
            downlink_scheduled = True
        sim.step()

    assert downlink_scheduled
    assert node.downlink_pending == 0
    assert node.fcnt_down == 1


def test_clean_channel_class_c_downlink_delivery():
    channel = make_clean_channel()
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        channels=[channel],
        node_class="C",
        fixed_sf=7,
        fixed_tx_power=14,
        seed=999,
    )

    node = sim.nodes[0]
    sim.network_server.send_downlink(node, b"hello")
    sim.run()

    assert node.downlink_pending == 0
    assert node.fcnt_down == 1
