"""Outils simples pour l'interférence (placeholder)."""


def aggregate_interference(powers_dbm: list[float]) -> float:
    """Agrège des puissances en dBm via une somme linéaire simplifiée."""
    linear = [10 ** (p / 10) for p in powers_dbm]
    total_linear = sum(linear)
    return 10 * __import__("math").log10(total_linear) if total_linear > 0 else float("-inf")
