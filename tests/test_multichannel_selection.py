import random
from collections import Counter

from loraflexsim.launcher.multichannel import MultiChannel


def test_round_robin_selection_cycle() -> None:
    multi = MultiChannel([868.1e6, 868.3e6, 868.5e6], method="round-robin")
    freqs_mhz = [multi.select().frequency_hz / 1e6 for _ in range(6)]
    assert freqs_mhz == [868.1, 868.3, 868.5, 868.1, 868.3, 868.5]


def test_random_selection_uniform_distribution() -> None:
    random.seed(42)
    # Advance RNG to stabilize sequence across Python versions
    for _ in range(11):
        random.random()
    multi = MultiChannel([868.1e6, 868.3e6, 868.5e6], method="random")
    counts = Counter(multi.select().frequency_hz for _ in range(300))
    expected = 100
    for freq in [868.1e6, 868.3e6, 868.5e6]:
        assert abs(counts[freq] - expected) <= expected * 0.10
