import math

from loraflexsim.launcher.simulator import _PowerTimeline


def test_power_timeline_average_power():
    timeline = _PowerTimeline()
    timeline.add(1, 0.0, 2.0, 2.0)
    timeline.add(2, 1.0, 3.0, 1.0)
    avg = timeline.average_power(0.0, 4.0, base_power=1.0)
    assert math.isclose(avg, 2.5)


def test_power_timeline_prune_and_changes():
    timeline = _PowerTimeline()
    timeline.add(1, 0.0, 1.0, 3.0)
    timeline.add(2, 0.5, 2.0, 2.0)
    timeline.prune(1.0)
    assert not timeline.is_empty()
    changes = timeline.power_changes(0.5, 2.0, base_power=1.0)
    assert changes == {0.5: 3.0, 2.0: -3.0}


