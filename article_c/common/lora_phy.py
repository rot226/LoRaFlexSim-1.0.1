"""Modèle physique LoRa avec hypothèses simplifiées."""

from __future__ import annotations

import math


SENSITIVITY_DBM_125KHZ = {
    7: -123.0,
    8: -126.0,
    9: -129.0,
    10: -132.0,
    11: -134.5,
    12: -137.0,
}
"""Sensibilité typique (dBm) par SF pour BW=125 kHz.

Hypothèse : valeurs indicatives issues de fiches techniques LoRa pour
un rapport signal/bruit nominal, sans marge d'implémentation spécifique.
"""


def toa_lora(sf: int, bw: int, cr: int, payload_bytes: int) -> float:
    """Calcule le Time-On-Air LoRa (ms) selon la formule Semtech simplifiée.

    Hypothèses :
    - Préambule de 8 symboles.
    - En-tête explicite (IH=0) et CRC activé (CRC=1).
    - L'optimisation bas débit (DE=1) est appliquée pour SF>=11 et BW<=125 kHz.
    - Le taux de codage ``cr`` est codé 1..4 pour 4/5..4/8.
    """

    if sf < 6 or sf > 12:
        raise ValueError("sf doit être entre 6 et 12")
    if bw <= 0:
        raise ValueError("bw doit être positif (kHz)")
    if cr < 1 or cr > 4:
        raise ValueError("cr doit être entre 1 et 4 (4/5..4/8)")
    if payload_bytes < 0:
        raise ValueError("payload_bytes doit être positif ou nul")

    bw_hz = bw * 1000
    ts = (2**sf) / bw_hz
    de = 1 if (sf >= 11 and bw <= 125) else 0
    ih = 0
    crc = 1
    preamble_symbols = 8

    payload_term = max(
        0,
        math.ceil(
            (8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih)
            / (4 * (sf - 2 * de))
        )
        * (cr + 4),
    )
    payload_symbols = 8 + payload_term
    total_symbols = preamble_symbols + 4.25 + payload_symbols
    return total_symbols * ts * 1000


def bitrate_lora(sf: int, bw: int, cr: int) -> float:
    """Retourne le débit utile LoRa (bps) pour un SF/BW/CR donné.

    Hypothèses :
    - ``cr`` est codé 1..4 pour 4/5..4/8.
    - Débit = SF * (4 / (4 + CR)) * BW / 2**SF.
    """

    if cr < 1 or cr > 4:
        raise ValueError("cr doit être entre 1 et 4 (4/5..4/8)")
    bw_hz = bw * 1000
    return sf * (4 / (4 + cr)) * bw_hz / (2**sf)


def coding_rate_to_cr(coding_rate: str) -> int:
    """Convertit un coding rate '4/5'..'4/8' en valeur CR 1..4."""

    parts = coding_rate.split("/")
    if len(parts) != 2 or parts[0] != "4":
        raise ValueError(f"coding_rate invalide: {coding_rate}")
    denominator = int(parts[1])
    cr_value = denominator - 4
    if cr_value < 1 or cr_value > 4:
        raise ValueError(f"coding_rate hors plage: {coding_rate}")
    return cr_value


def compute_airtime(payload_bytes: int, sf: int, bw_khz: int, cr: int = 1) -> float:
    """Calcule un airtime LoRa en millisecondes via ``toa_lora``.

    Hypothèses : cf. ``toa_lora``.
    """

    return toa_lora(sf=sf, bw=bw_khz, cr=cr, payload_bytes=payload_bytes)
