import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402


def test_cost231_path_loss_vs_log_distance():
    adv = AdvancedChannel()
    pl = adv.path_loss(1000)
    assert pl > 100


def test_rayleigh_fading_variability():
    adv = AdvancedChannel(fading="rayleigh")
    r1, _ = adv.compute_rssi(14.0, 100.0)
    r2, _ = adv.compute_rssi(14.0, 100.0)
    assert r1 != r2
