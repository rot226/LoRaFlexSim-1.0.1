import math
from loraflexsim.launcher.channel import Channel


def test_channel_sensitivity_values():
    channel = Channel()
    noise = -174 + 10 * math.log10(channel.bandwidth) + channel.noise_figure_dB
    expected = {sf: noise + snr for sf, snr in Channel.SNR_THRESHOLDS.items()}
    for sf, th in expected.items():
        assert math.isclose(channel.sensitivity_dBm[sf], th, abs_tol=0.1)
