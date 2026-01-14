"""Calcul de métriques pour les simulations."""

from collections.abc import Iterable


def packet_delivery_ratio(rx_success: int, tx_total: int) -> float:
    """Calcule le PDR."""
    return rx_success / tx_total if tx_total else 0.0


def goodput_bps(payload_bits_success: int, duration_s: float) -> float:
    """Calcule le goodput en bps."""
    return payload_bits_success / duration_s if duration_s else 0.0


def collision_rate(collisions: int, tx_total: int) -> float:
    """Calcule le taux de collision."""
    return collisions / tx_total if tx_total else 0.0


def outage_probability(outage_events: int, tx_total: int) -> float:
    """Calcule la probabilité d'outage."""
    return outage_events / tx_total if tx_total else 0.0


def energy_per_success_bit(
    airtime_ms: Iterable[float],
    payload_bits_success: int,
    tx_power_dbm: float,
) -> float:
    """Calcule l'énergie par bit correctement reçu via le ToA."""
    total_airtime_s = sum(airtime_ms) / 1000.0
    power_w = 10 ** ((tx_power_dbm - 30) / 10)
    total_energy = total_airtime_s * power_w
    return total_energy / payload_bits_success if payload_bits_success else 0.0
