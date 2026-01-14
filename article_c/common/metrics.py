"""Calcul de métriques (placeholder)."""


def packet_delivery_ratio(received: int, sent: int) -> float:
    """Calcule le PDR."""
    return received / sent if sent else 0.0
