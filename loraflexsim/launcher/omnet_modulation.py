"""Helpers for BER/SER calculations using analytical LoRa expressions."""

from __future__ import annotations

import math


def calculate_ber(snir: float, sf: int) -> float:
    r"""Return BER using Croce et al. (2018) approximation.

    The expression models the bit error probability of LoRa in AWGN as
    :math:`0.5\,\mathrm{erfc}\left(\sqrt{\mathrm{SNR}\,2^{SF}/(2\pi)}\right)`.
    ``snir`` is the linear signal-to-noise ratio.
    """

    n = 2 ** sf
    arg = math.sqrt(snir * n / (2.0 * math.pi))
    ber = 0.5 * math.erfc(arg)
    return min(max(ber, 0.0), 1.0)


def calculate_ber_flora(snir: float, sf: int, bandwidth: float) -> float:
    """Return BER using the legacy FLoRa implementation.

    The algorithm is a direct transcription of
    ``LoRaModulation::calculateBER`` from the C++ FLoRa code base.  It is
    based on the IEEE 802.15.4 OQPSK approximation and relies on the
    ``bandwidth`` and the bitrate associated with ``sf``.  The bitrate is
    estimated from the LoRa specification as ``sf * bandwidth / 2**sf`` (no
    coding rate is considered).  ``snir`` is the linear signal-to-noise
    ratio.
    """

    # Reconstruct bitrate the same way as in FLoRa.  Using CR = 1 yields the
    # expression below.
    bitrate = sf * bandwidth / (2 ** sf)
    dsnr = 20.0 * snir * bandwidth / bitrate
    d_sum = 0.0

    for k in range(2, 8, 2):  # k = 2,4,6 with symmetric counterparts
        term = math.exp(dsnr * (1.0 / k - 1.0))
        term_sym = math.exp(dsnr * (1.0 / (16 - k) - 1.0))
        d_sum += math.comb(16, k) * (term + term_sym)

    k = 8
    d_sum += math.comb(16, k) * math.exp(dsnr * (1.0 / k - 1.0))

    for k in range(3, 8, 2):  # k = 3,5,7 and symmetric counterparts
        term = math.exp(dsnr * (1.0 / k - 1.0))
        term_sym = math.exp(dsnr * (1.0 / (16 - k) - 1.0))
        d_sum -= math.comb(16, k) * (term + term_sym)

    k = 15
    d_sum -= math.comb(16, k) * math.exp(dsnr * (1.0 / k - 1.0))

    k = 16
    d_sum += math.comb(16, k) * math.exp(dsnr * (1.0 / k - 1.0))

    ber = (8.0 / 15.0) * (1.0 / 16.0) * d_sum
    return min(max(ber, 0.0), 1.0)


def calculate_ser(snir: float, sf: int) -> float:
    """Return SER from BER for an ``sf``-bit LoRa symbol."""

    ber = calculate_ber(snir, sf)
    ser = 1.0 - (1.0 - ber) ** sf
    return min(max(ser, 0.0), 1.0)
