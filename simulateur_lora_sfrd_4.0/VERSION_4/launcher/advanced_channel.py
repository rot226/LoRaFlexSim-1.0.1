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
        terrain: str = "urban",
        weather_loss_dB_per_km: float = 0.0,
        **kwargs,
    ) -> None:
        from .channel import Channel

        self.base = Channel(**kwargs)
        self.base_station_height = base_station_height
        self.mobile_height = mobile_height
        self.propagation_model = propagation_model
        self.fading = fading
        self.terrain = terrain.lower()
        self.weather_loss_dB_per_km = weather_loss_dB_per_km

    # ------------------------------------------------------------------
    # Propagation models
    # ------------------------------------------------------------------
    def path_loss(self, distance: float) -> float:
        """Return path loss in dB for the selected model."""
        d = distance
        if self.propagation_model == "3d":
            d = math.sqrt(distance ** 2 + (self.base_station_height - self.mobile_height) ** 2)
            loss = self.base.path_loss(d)
        elif self.propagation_model == "cost231":
            loss = self._cost231_loss(distance)
        elif self.propagation_model == "okumura_hata":
            loss = self._okumura_hata_loss(distance)
        else:
            loss = self.base.path_loss(distance)

        if self.weather_loss_dB_per_km:
            loss += self.weather_loss_dB_per_km * (max(d, 1.0) / 1000.0)
        return loss

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

    def _okumura_hata_loss(self, distance: float) -> float:
        distance_km = max(distance / 1000.0, 1e-3)
        freq_mhz = self.base.frequency_hz / 1e6
        hb = self.base_station_height
        hm = self.mobile_height
        a_hm = (
            (1.1 * math.log10(freq_mhz) - 0.7) * hm
            - (1.56 * math.log10(freq_mhz) - 0.8)
        )
        pl = (
            69.55
            + 26.16 * math.log10(freq_mhz)
            - 13.82 * math.log10(hb)
            - a_hm
            + (44.9 - 6.55 * math.log10(hb)) * math.log10(distance_km)
        )
        if self.terrain == "suburban":
            pl -= 2 * (math.log10(freq_mhz / 28.0)) ** 2 - 5.4
        elif self.terrain == "open":
            pl -= 4.78 * (math.log10(freq_mhz)) ** 2 - 18.33 * math.log10(freq_mhz) + 40.94
        return pl

    # ------------------------------------------------------------------
    def compute_rssi(self, tx_power_dBm: float, distance: float) -> tuple[float, float]:
        """Return RSSI and SNR for the advanced channel."""
        loss = self.path_loss(distance)
        if self.base.shadowing_std > 0:
            loss += random.gauss(0, self.base.shadowing_std)

        rssi = tx_power_dBm - loss - self.base.cable_loss_dB
        if self.base.tx_power_std > 0:
            rssi += random.gauss(0, self.base.tx_power_std)
        if self.base.fast_fading_std > 0:
            rssi += random.gauss(0, self.base.fast_fading_std)

        noise = self.base.noise_floor_dBm()

        if self.fading == "rayleigh":
            u = random.random()
            rayleigh = math.sqrt(-2.0 * math.log(max(u, 1e-12)))
            rssi += 20 * math.log10(rayleigh)

        return rssi, rssi - noise
