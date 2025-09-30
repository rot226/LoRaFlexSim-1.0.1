import math
from loraflexsim.launcher.propagation_models import LogDistanceShadowing
from loraflexsim.launcher.channel import Channel


def test_logdistance_urban_preset():
    model = LogDistanceShadowing(environment="urban")
    model.shadowing_std = 0.0
    loss = model.path_loss(80.0)
    expected = 127.41 + 10 * 2.08 * math.log10(80.0 / 40.0)
    assert math.isclose(loss, expected, rel_tol=1e-6)


def test_channel_preset_matches_logdistance():
    ch = Channel(environment="urban")
    ch.shadowing_std = 0.0
    ch.fast_fading_std = 0.0
    ld = LogDistanceShadowing(environment="urban")
    ld.shadowing_std = 0.0
    assert math.isclose(ch.path_loss(80.0), ld.path_loss(80.0), rel_tol=1e-6)
