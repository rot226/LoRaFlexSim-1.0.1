"""Simplified OMNeT++ physical layer helpers."""

from __future__ import annotations

import math
import random

from .omnet_model import OmnetModel


class OmnetPHY:
    """Replicate OMNeT++ FLoRa PHY calculations."""

    def __init__(self, channel) -> None:
        self.channel = channel
        self.model = OmnetModel(
            channel.fine_fading_std,
            channel.omnet.correlation,
            channel.omnet.noise_std,
        )

    # ------------------------------------------------------------------
    def path_loss(self, distance: float) -> float:
        """Return path loss in dB using the log distance model."""
        if distance <= 0:
            return 0.0
        freq_mhz = self.channel.frequency_hz / 1e6
        pl_d0 = 32.45 + 20 * math.log10(freq_mhz) - 60.0
        loss = pl_d0 + 10 * self.channel.path_loss_exp * math.log10(max(distance, 1.0))
        return loss + self.channel.system_loss_dB

    def noise_floor(self) -> float:
        """Return the noise floor (dBm) including optional variations."""
        ch = self.channel
        thermal = ch.receiver_noise_floor_dBm + 10 * math.log10(ch.bandwidth)
        noise = thermal + ch.noise_figure_dB + ch.interference_dB
        if ch.noise_floor_std > 0:
            noise += random.gauss(0.0, ch.noise_floor_std)
        noise += self.model.noise_variation()
        return noise

    def compute_rssi(
        self,
        tx_power_dBm: float,
        distance: float,
        sf: int | None = None,
        *,
        freq_offset_hz: float | None = None,
        sync_offset_s: float | None = None,
    ) -> tuple[float, float]:
        ch = self.channel
        loss = self.path_loss(distance)
        if ch.shadowing_std > 0:
            loss += random.gauss(0.0, ch.shadowing_std)
        rssi = (
            tx_power_dBm
            + ch.tx_antenna_gain_dB
            + ch.rx_antenna_gain_dB
            - loss
            - ch.cable_loss_dB
        )
        if ch.tx_power_std > 0:
            rssi += random.gauss(0.0, ch.tx_power_std)
        if ch.fast_fading_std > 0:
            rssi += random.gauss(0.0, ch.fast_fading_std)
        if ch.time_variation_std > 0:
            rssi += random.gauss(0.0, ch.time_variation_std)
        rssi += self.model.fine_fading()
        rssi += ch.rssi_offset_dB
        snr = rssi - self.noise_floor() + ch.snr_offset_dB
        if sf is not None:
            snr += 10 * math.log10(2 ** sf)
        return rssi, snr

    def capture(self, rssi_list: list[float]) -> list[bool]:
        """Return list of booleans indicating which signals are captured."""
        if not rssi_list:
            return []
        order = sorted(range(len(rssi_list)), key=lambda i: rssi_list[i], reverse=True)
        winners = [False] * len(rssi_list)
        if len(order) == 1:
            winners[order[0]] = True
            return winners
        if rssi_list[order[0]] - rssi_list[order[1]] >= self.channel.capture_threshold_dB:
            winners[order[0]] = True
        return winners

