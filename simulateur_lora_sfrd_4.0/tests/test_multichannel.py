import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.multichannel import MultiChannel  # noqa: E402


def test_round_robin_cycles_sequentially():
    freqs = [868.1e6, 868.3e6, 868.5e6]
    mc = MultiChannel(freqs)
    results = [mc.select().frequency_hz for _ in range(6)]
    assert results == freqs + freqs


def test_random_select_mask_returns_only_allowed():
    freqs = [868.1e6, 868.3e6, 868.5e6]
    mc = MultiChannel(freqs, method="random")
    mask = 0b101  # allow channels 0 and 2
    random.seed(42)
    for _ in range(20):
        ch = mc.select_mask(mask)
        assert ch.frequency_hz in {freqs[0], freqs[2]}

