"""Modèle de propagation (placeholder)."""


def free_space_path_loss(distance_km: float, freq_mhz: float) -> float:
    """Retourne une perte en dB via le modèle de l'espace libre."""
    if distance_km <= 0:
        raise ValueError("distance_km doit être positive")
    return 32.45 + 20 * __import__("math").log10(distance_km) + 20 * __import__("math").log10(freq_mhz)
