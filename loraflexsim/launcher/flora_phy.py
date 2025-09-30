"""FLoRa-like physical layer helpers."""

from __future__ import annotations

import math
import random

from .omnet_modulation import calculate_ber, calculate_ser


class FloraPHY:
    """Replicate FLoRa path loss and capture formulas.

    ``loss_model`` selects which path loss equations to use:

    - ``"lognorm"`` (default): original FLoRa log-normal shadowing
      ``PATH_LOSS_D0`` at ``REFERENCE_DISTANCE`` with exponent ``path_loss_exp``.
    - ``"oulu"``: constants from ``LoRaPathLossOulu``.
    - ``"hata"``: constants from ``LoRaHataOkumura``.
    """

    NON_ORTH_DELTA = [
        [1, -8, -9, -9, -9, -9],
        [-11, 1, -11, -12, -13, -13],
        [-15, -13, 1, -13, -14, -15],
        [-19, -18, -17, 1, -17, -18],
        [-22, -22, -21, -20, 1, -20],
        [-25, -25, -25, -24, -23, 1],
    ]

    REFERENCE_DISTANCE = 40.0
    PATH_LOSS_D0 = 127.41
    OULU_D0 = 1000.0
    OULU_N = 2.32
    OULU_B = 128.95
    OULU_ANTENNA_GAIN = 2.0
    HATA_K1 = 127.5
    HATA_K2 = 35.2
    SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

    def __init__(self, channel, loss_model: str = "lognorm") -> None:
        self.channel = channel
        self.loss_model = loss_model

    def path_loss(self, distance: float) -> float:
        if distance <= 0:
            return 0.0
        d = max(distance, 1.0)
        if self.loss_model == "oulu":
            loss = (
                self.OULU_B
                + 10 * self.OULU_N * math.log10(d / self.OULU_D0)
                - self.OULU_ANTENNA_GAIN
            )
        elif self.loss_model == "hata":
            loss = self.HATA_K1 + self.HATA_K2 * math.log10(d / 1000.0)
        else:
            loss = (
                self.PATH_LOSS_D0
                + 10
                * self.channel.path_loss_exp
                * math.log10(d / self.REFERENCE_DISTANCE)
            )
        if self.channel.shadowing_std > 0:
            loss += random.gauss(0.0, self.channel.shadowing_std)
        return loss + self.channel.system_loss_dB

    def capture(
        self,
        rssi_list: list[float],
        sf_list: list[int],
        start_list: list[float],
        end_list: list[float],
        freq_list: list[float],
    ) -> list[bool]:
        """Return the capture decision for each concurrent signal.

        The algorithm follows the same logic as ``LoRaReceiver`` in FLoRa:
        the strongest packet wins only if the power difference with each
        interferer is above the ``NON_ORTH_DELTA`` threshold *and* the
        interferer overlaps past the ``preamble - capture_window_symbols``
        window (six symbols in the FLoRa configuration).
        """

        if not rssi_list:
            return []

        order = sorted(range(len(rssi_list)), key=lambda i: rssi_list[i], reverse=True)
        winners = [False] * len(rssi_list)
        if len(order) == 1:
            winners[order[0]] = True
            return winners

        idx0 = order[0]
        sf0 = sf_list[idx0]
        rssi0 = rssi_list[idx0]
        freq0 = freq_list[idx0]
        start0 = start_list[idx0]
        end0 = end_list[idx0]

        symbol_time = (2 ** sf0) / self.channel.bandwidth
        cs_begin = start0 + symbol_time * (
            self.channel.preamble_symbols - self.channel.capture_window_symbols
        )

        captured = True
        for idx in order[1:]:
            # only interfering packets on the same frequency matter
            if freq_list[idx] != freq0:
                continue

            # check if the packets overlap in time
            overlap = min(end0, end_list[idx]) > max(start0, start_list[idx])
            if not overlap:
                continue

            diff = rssi0 - rssi_list[idx]
            th = self.NON_ORTH_DELTA[sf0 - 7][sf_list[idx] - 7]
            capture_effect = diff >= th

            timing_collision = cs_begin < end_list[idx]

            if not capture_effect and timing_collision:
                captured = False
                break

        if captured:
            winners[idx0] = True

        return winners

    def packet_error_rate(
        self, snr: float, sf: int, payload_bytes: int = 20, per_model: str = "croce"
    ) -> float:
        """Return PER using either a logistic or Croce et al. model.

        Parameters
        ----------
        snr: float
            Signal-to-noise ratio in dB.
        sf: int
            Spreading factor of the signal.
        payload_bytes: int
            Payload length in bytes. Only used by the ``croce`` model.
        per_model: str
            ``"logistic"`` to use the FLoRa-style logistic approximation or
            ``"croce"`` to rely on BER/SER expressions from Croce et al.
        """

        if per_model == "none":
            return 0.0
        if per_model == "logistic":
            th = self.SNR_THRESHOLDS[sf]
            per = 1.0 / (1.0 + math.exp(2 * (snr - (th + 2))))
            return min(max(per, 0.0), 1.0)

        snir = 10 ** (snr / 10.0)
        ber = calculate_ber(snir, sf)
        ser = calculate_ser(snir, sf)
        n_bits = payload_bytes * 8
        per_bit = 1.0 - (1.0 - ber) ** n_bits
        n_sym = math.ceil(n_bits / sf)
        per_sym = 1.0 - (1.0 - ser) ** n_sym
        per = max(per_bit, per_sym)
        return min(max(per, 0.0), 1.0)
