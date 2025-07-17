"""Advanced physical layer models for LoRa simulations."""

from __future__ import annotations

import math
import random


class AdvancedChannel:
    """Optional channel with more detailed propagation models."""

    def __init__(
        self,
        base_station_height: float = 30.0,
        mobile_height: float = 1.5,
        propagation_model: str = "cost231",
        fading: str = "rayleigh",
        **kwargs,
    ) -> None:
        from .channel import Channel

        self.base = Channel(**kwargs)
        self.base_station_height = base_station_height
        self.mobile_height = mobile_height
        self.propagation_model = propagation_model
        self.fading = fading

    # ------------------------------------------------------------------
    # Propagation models
    # ------------------------------------------------------------------
    def path_loss(self, distance: float) -> float:
        """Return path loss in dB for the selected model."""
        if self.propagation_model == "cost231":
            return self._cost231_loss(distance)
        return self.base.path_loss(distance)

    def _cost231_loss(self, distance: float) -> float:
        distance_km = max(distance / 1000.0, 1e-3)
        freq_mhz = self.base.frequency_hz / 1e6
        a_hm = (
            (1.1 * math.log10(freq_mhz) - 0.7) * self.mobile_height
            - (1.56 * math.log10(freq_mhz) - 0.8)
        )
        return (
            46.3
            + 33.9 * math.log10(freq_mhz)
            - 13.82 * math.log10(self.base_station_height)
            - a_hm
            + (44.9 - 6.55 * math.log10(self.base_station_height))
            * math.log10(distance_km)
        )

    # ------------------------------------------------------------------
    def compute_rssi(self, tx_power_dBm: float, distance: float) -> tuple[float, float]:
        """Return RSSI and SNR for the advanced channel."""
        base_rssi, base_snr = self.base.compute_rssi(tx_power_dBm, distance)
        noise = base_rssi - base_snr

        rssi = base_rssi
        if self.fading == "rayleigh":
            u = random.random()
            rayleigh = math.sqrt(-2.0 * math.log(max(u, 1e-12)))
            rssi += 20 * math.log10(rayleigh)

        return rssi, rssi - noise
