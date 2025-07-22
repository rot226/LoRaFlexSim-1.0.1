"""Extra physical layer features inspired by the OMNeT++ FLoRa model."""

from __future__ import annotations

import math
import random


class OmnetModel:
    """Handle fine fading, oscillator drift and thermal noise."""

    def __init__(
        self,
        fading_std: float = 0.0,
        correlation: float = 0.9,
        noise_std: float = 0.0,
        *,
        freq_drift_std: float = 0.0,
        clock_drift_std: float = 0.0,
        temperature_K: float = 290.0,
    ) -> None:
        self.fading_std = fading_std
        self.correlation = correlation
        self.noise_std = noise_std
        self.freq_drift_std = freq_drift_std
        self.clock_drift_std = clock_drift_std
        self.temperature_K = temperature_K
        self._fading = 0.0
        self._noise = 0.0
        self._freq = 0.0
        self._clock = 0.0

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

    def frequency_drift(self) -> float:
        """Return a correlated frequency drift sample (Hz)."""
        if self.freq_drift_std <= 0.0:
            return 0.0
        gaussian = random.gauss(0.0, self.freq_drift_std)
        self._freq = self.correlation * self._freq + (1 - self.correlation) * gaussian
        return self._freq

    def clock_drift(self) -> float:
        """Return a correlated clock drift sample (s)."""
        if self.clock_drift_std <= 0.0:
            return 0.0
        gaussian = random.gauss(0.0, self.clock_drift_std)
        self._clock = self.correlation * self._clock + (1 - self.correlation) * gaussian
        return self._clock

    def thermal_noise_dBm(self, bandwidth: float) -> float:
        """Return the thermal noise level for the given bandwidth."""
        k = 1.38064852e-23
        noise_w = k * self.temperature_K * bandwidth
        return 10 * math.log10(noise_w) + 30

    def variable_thermal_noise_dBm(self, bandwidth: float) -> float:
        """Return the thermal noise including slow variations."""
        base = self.thermal_noise_dBm(bandwidth)
        return base + self.noise_variation()

