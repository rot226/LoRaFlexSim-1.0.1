"""Modèle physique LoRa (placeholder)."""


def compute_airtime(payload_bytes: int, sf: int, bw_khz: int) -> float:
    """Calcule un airtime simplifié en millisecondes."""
    symbol_rate = bw_khz * 1000 / (2 ** sf)
    return (payload_bytes * 8) / symbol_rate * 1000
