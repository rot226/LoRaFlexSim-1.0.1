import sys
from pathlib import Path
import random
import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402
from VERSION_4.launcher.compare_flora import load_flora_rx_stats  # noqa: E402


def test_advanced_channel_matches_flora():
    random.seed(0)
    adv = AdvancedChannel(
        propagation_model="",
        fading="",
        shadowing_std=0,
    )
    rssi, snr = adv.compute_rssi(14.0, 99.2, sf=7)
    flora_csv = Path(__file__).parent / "data" / "flora_rx_stats.csv"
    flora = load_flora_rx_stats(flora_csv)
    assert rssi == pytest.approx(flora["rssi"], abs=1e-3)
    assert snr == pytest.approx(flora["snr"], abs=1e-1)
