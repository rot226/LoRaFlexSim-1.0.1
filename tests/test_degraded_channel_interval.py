from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.adr_standard_1 import apply as apply_adr
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.multichannel import MultiChannel
from loraflexsim.launcher.advanced_channel import AdvancedChannel


def test_interval_with_degraded_channel():
    mean_interval = 2.0
    packets = 1000
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=mean_interval,
        packets_to_send=packets,
        pure_poisson_mode=True,
        mobility=False,
        seed=1,
    )
    apply_adr(sim, degrade_channel=True, profile="flora", capture_mode="flora")
    sim.run()
    node = sim.nodes[0]
    average = node._last_arrival_time / node.packets_sent
    assert node.packets_sent == packets
    assert abs(average - mean_interval) / mean_interval < 0.02


def test_channels_identical_after_degrade():
    channels = [
        Channel(frequency_hz=868.1e6),
        Channel(frequency_hz=868.3e6),
        Channel(frequency_hz=868.5e6),
    ]
    multi = MultiChannel(channels)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=1.0,
        packets_to_send=1,
        pure_poisson_mode=True,
        mobility=False,
        channels=multi,
        seed=1,
    )
    apply_adr(sim, degrade_channel=True, profile="flora", capture_mode="flora")

    attrs = [
        "path_loss_exp",
        "shadowing_std",
        "detection_threshold_dBm",
        "capture_threshold_dB",
        "flora_capture",
        "flora_loss_model",
    ]
    reference = sim.multichannel.channels[0]
    ple, shad, *_ = Channel.ENV_PRESETS["flora"]
    assert reference.path_loss_exp == ple
    assert reference.shadowing_std == shad
    assert reference.flora_capture is True
    assert reference.flora_loss_model == "lognorm"
    for ch in sim.multichannel.channels[1:]:
        for attr in attrs:
            assert getattr(ch, attr) == getattr(reference, attr)


def test_custom_profile_changes_params():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=1.0,
        packets_to_send=1,
        pure_poisson_mode=True,
        mobility=False,
        seed=1,
    )
    apply_adr(sim, degrade_channel=True, profile="urban")
    ple, shad, *_ = Channel.ENV_PRESETS["urban"]
    ch = sim.multichannel.channels[0]
    assert ch.path_loss_exp == ple
    assert ch.shadowing_std == shad


def test_degraded_channel_reduces_pdr():
    sim_ideal = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=4000,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=10,
        mobility=False,
        fixed_sf=7,
        seed=0,
    )
    apply_adr(sim_ideal)
    sim_ideal.run()
    pdr_ideal = sim_ideal.get_metrics()["PDR"]

    sim_degraded = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=4000,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=10,
        mobility=False,
        fixed_sf=7,
        seed=0,
    )
    apply_adr(sim_degraded, degrade_channel=True, profile="flora")
    node = sim_degraded.nodes[0]
    assert isinstance(node.channel, AdvancedChannel)
    sim_degraded.run()
    pdr_degraded = sim_degraded.get_metrics()["PDR"]
    assert pdr_degraded < pdr_ideal
