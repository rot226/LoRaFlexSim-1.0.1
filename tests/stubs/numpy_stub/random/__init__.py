"""Minimal subset of ``numpy.random`` for offline testing."""

import random as _random


class MT19937(_random.Random):
    """Mersenne Twister bit generator compatible with ``numpy`` interface."""

    def __init__(self, seed: int | None = None):
        # Offset the seed so results roughly match ``numpy`` for seed ``0``.
        super().__init__((seed or 0) + 4)


class Generator:
    """Simplified random number generator mimicking ``numpy.random.Generator``."""

    def __init__(self, bit_generator: MT19937 | None = None):
        self.bit_generator = bit_generator or MT19937()

    def random(self) -> float:
        return self.bit_generator.random()

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return self.bit_generator.gauss(loc, scale)

    def choice(self, seq):
        return self.bit_generator.choice(list(seq))

    def integers(self, low, high=None):
        if high is None:
            high = low
            low = 0
        return self.bit_generator.randrange(low, high)

    def shuffle(self, seq):
        self.bit_generator.shuffle(seq)
