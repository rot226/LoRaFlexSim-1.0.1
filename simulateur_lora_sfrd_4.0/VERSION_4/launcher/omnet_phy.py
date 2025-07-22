"""Simplified OMNeT++ physical layer helpers."""

from __future__ import annotations

import math
import random

from .omnet_model import OmnetModel


class _CorrelatedValue:
    """Simple correlated random walk."""

    def __init__(self, mean: float, std: float, correlation: float) -> None:
        self.mean = mean
        self.std = std
        self.corr = correlation
        self.value = mean

    def sample(self) -> float:
        self.value = self.corr * self.value + (1.0 - self.corr) * self.mean
        if self.std > 0.0:
            self.value += random.gauss(0.0, self.std)
        return self.value


class OmnetPHY:
    """Replicate OMNeT++ FLoRa PHY calculations with extra impairments."""

    def __init__(
        self,
        channel,
        *,
        dev_frequency_offset_hz: float = 0.0,
        dev_freq_offset_std_hz: float = 0.0,
        temperature_std_K: float = 0.0,
        pa_non_linearity_dB: float = 0.0,
        pa_non_linearity_std_dB: float = 0.0,
        phase_noise_std_dB: float = 0.0,
        oscillator_leakage_dB: float = 0.0,
        oscillator_leakage_std_dB: float = 0.0,
        rx_fault_std_dB: float = 0.0,
    ) -> None:
        """Initialise helper with optional hardware impairments."""
        self.channel = channel
        self.model = OmnetModel(
            channel.fine_fading_std,
            channel.omnet.correlation,
            channel.omnet.noise_std,
            freq_drift_std=channel.omnet.freq_drift_std,
            clock_drift_std=channel.omnet.clock_drift_std,
            temperature_K=channel.omnet.temperature_K,
        )
        corr = channel.omnet.correlation
        self._dev_freq = _CorrelatedValue(
            dev_frequency_offset_hz,
            dev_freq_offset_std_hz,
            corr,
        )
        self._temperature = _CorrelatedValue(
            channel.omnet.temperature_K,
            temperature_std_K,
            corr,
        )
        self._pa_nl = _CorrelatedValue(
            pa_non_linearity_dB,
            pa_non_linearity_std_dB,
            corr,
        )
        self._phase_noise = _CorrelatedValue(0.0, phase_noise_std_dB, corr)
        self._osc_leak = _CorrelatedValue(
            oscillator_leakage_dB,
            oscillator_leakage_std_dB,
            corr,
        )
        self._rx_fault = _CorrelatedValue(0.0, rx_fault_std_dB, corr)

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
        if self._temperature.std > 0.0:
            temp = self._temperature.sample()
            original = self.model.temperature_K
            self.model.temperature_K = temp
            thermal = self.model.thermal_noise_dBm(ch.bandwidth)
            self.model.temperature_K = original
        else:
            thermal = self.model.thermal_noise_dBm(ch.bandwidth)
        noise = thermal + ch.noise_figure_dB + ch.interference_dB
        if ch.humidity_noise_coeff_dB != 0.0:
            noise += ch.humidity_noise_coeff_dB * (ch._humidity.sample() / 100.0)
        for f, bw, power in ch.band_interference:
            half = (bw + ch.bandwidth) / 2.0
            if abs(ch.frequency_hz - f) <= half:
                noise += power
        if ch.noise_floor_std > 0:
            noise += random.gauss(0.0, ch.noise_floor_std)
        noise += self.model.noise_variation()
        noise += self._osc_leak.sample()
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

        if freq_offset_hz is None:
            freq_offset_hz = ch.frequency_offset_hz
        # Include time-varying frequency drift
        freq_offset_hz += self.model.frequency_drift()
        freq_offset_hz += self._dev_freq.sample()
        if sync_offset_s is None:
            sync_offset_s = ch.sync_offset_s
        # Include short-term clock jitter
        sync_offset_s += self.model.clock_drift()

        tx_power_dBm += self._pa_nl.sample()
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
        rssi -= ch._filter_attenuation_db(freq_offset_hz)

        snr = rssi - self.noise_floor() + ch.snr_offset_dB
        penalty = self._alignment_penalty_db(freq_offset_hz, sync_offset_s, sf)
        snr -= penalty
        snr -= abs(self._phase_noise.sample())
        snr -= abs(self._rx_fault.sample())
        if sf is not None:
            snr += 10 * math.log10(2 ** sf)
        return rssi, snr

    def _alignment_penalty_db(
        self, freq_offset_hz: float, sync_offset_s: float, sf: int | None
    ) -> float:
        """Return SNR penalty for imperfect alignment."""
        bw = self.channel.bandwidth
        freq_factor = abs(freq_offset_hz) / (bw / 2.0)
        if sf is not None:
            symbol_time = (2 ** sf) / bw
        else:
            symbol_time = 1.0 / bw
        time_factor = abs(sync_offset_s) / symbol_time
        if freq_factor >= 1.0 and time_factor >= 1.0:
            return float("inf")
        return 10 * math.log10(1.0 + freq_factor ** 2 + time_factor ** 2)

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

