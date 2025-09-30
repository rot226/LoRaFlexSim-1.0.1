import random
import math
from loraflexsim.launcher.channel import Channel


def test_multipath_fading_mean():
    random.seed(0)
    ch = Channel(multipath_taps=3)
    vals = [ch._multipath_fading_db() for _ in range(1000)]
    avg = sum(vals) / len(vals)
    # Expect around 0.5 dB mean for Rayleigh fading
    assert 0.0 <= avg <= 1.0


def test_sensitivity_matches_theory():
    ch = Channel()
    bw = ch.bandwidth
    nf = ch.noise_figure_dB
    snr_req = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
    noise_floor = -174 + 10 * math.log10(bw) + nf
    for sf, req in snr_req.items():
        expected = noise_floor + req
        assert math.isclose(ch.sensitivity_dBm[sf], expected, abs_tol=0.1)
