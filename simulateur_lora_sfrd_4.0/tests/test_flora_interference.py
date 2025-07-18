import sys
import random
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402
from VERSION_4.launcher.compare_flora import load_flora_rx_stats  # noqa: E402


def test_interference_matches_flora():
    random.seed(0)
    ch = AdvancedChannel(
        propagation_model="",
        fading="rayleigh",
        shadowing_std=0,
        frequency_offset_hz=5000,
        sync_offset_s=0.001,
        multipath_paths=2,
    )
    rssi, snr = ch.compute_rssi(14.0, 100.0, sf=7)
    flora_csv = Path(__file__).parent / "data" / "flora_interference.csv"
    flora = load_flora_rx_stats(flora_csv)
    assert rssi == pytest.approx(flora["rssi"], abs=1e-3)
    assert snr == pytest.approx(flora["snr"], abs=1e-3)
