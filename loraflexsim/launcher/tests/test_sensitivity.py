import math

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.simulator import Simulator


def test_channel_sensitivity_values():
    channel = Channel()
    noise = -174 + 10 * math.log10(channel.bandwidth) + channel.noise_figure_dB
    expected = {sf: noise + snr for sf, snr in Channel.SNR_THRESHOLDS.items()}
    for sf, th in expected.items():
        assert math.isclose(channel.sensitivity_dBm[sf], th, abs_tol=0.1)


def test_channel_overrides_and_rng_seed(tmp_path):
    cfg = tmp_path / "radio.ini"
    cfg.write_text(
        """
        [channel]
        snir_fading_std = 2.5
        noise_floor_std = 0.75
        sensitivity_margin_dB = 1.5
        capture_threshold_dB = 7.0
        marginal_snir_margin_db = 0.9
        marginal_snir_drop_prob = 0.4
        snir_penalty_strength = 0.35
        interference_dB = 0.25
        """
    )

    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        mobility=False,
        packet_interval=1.0,
        packets_to_send=0,
        channel_config=cfg,
        seed=1234,
    )

    channel = sim.multichannel.channels[0]
    stream = sim.rng_manager.get_stream("channel", 0)

    assert channel.rng is stream
    assert math.isclose(channel.snir_fading_std, 2.5)
    assert math.isclose(channel.noise_floor_std, 0.75)
    assert math.isclose(channel.interference_dB, 0.25)
    assert math.isclose(channel.capture_threshold_dB, 7.0)
    assert math.isclose(channel.marginal_snir_margin_db, 0.9)
    assert math.isclose(channel.marginal_snir_drop_prob, 0.4)
    assert math.isclose(channel.snir_penalty_strength, 0.35)

    baseline_noise = -174 + 10 * math.log10(channel.bandwidth) + channel.noise_figure_dB
    expected_threshold = baseline_noise + channel.SNR_THRESHOLDS[7] + 1.5
    assert math.isclose(channel.sensitivity_dBm[7], expected_threshold, abs_tol=1e-6)
