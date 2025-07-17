"""Extra physical layer features inspired by the OMNeT++ FLoRa model."""

from __future__ import annotations

import math
import random


class OmnetModel:
    """Handle fine fading and variable thermal noise."""

    def __init__(self, fading_std: float = 0.0, correlation: float = 0.9, noise_std: float = 0.0) -> None:
        self.fading_std = fading_std
        self.correlation = correlation
        self.noise_std = noise_std
        self._fading = 0.0
        self._noise = 0.0

    def fine_fading(self) -> float:
        """Return a temporally correlated fading term (dB)."""
        if self.fading_std <= 0.0:
            return 0.0
        gaussian = random.gauss(0.0, self.fading_std)
        self._fading = self.correlation * self._fading + (1 - self.correlation) * gaussian
        return self._fading

    def noise_variation(self) -> float:
        """Return a temporally correlated noise variation (dB)."""
        if self.noise_std <= 0.0:
            return 0.0
        gaussian = random.gauss(0.0, self.noise_std)
        self._noise = self.correlation * self._noise + (1 - self.correlation) * gaussian
        return self._noise

