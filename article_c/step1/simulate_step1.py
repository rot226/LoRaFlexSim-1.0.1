"""Simulation de l'étape 1 (placeholder)."""

from dataclasses import dataclass
import random

from article_c.common.metrics import packet_delivery_ratio


@dataclass
class Step1Result:
    sent: int
    received: int

    @property
    def pdr(self) -> float:
        return packet_delivery_ratio(self.received, self.sent)


def run_simulation(
    density: float,
    algo: str,
    snir_mode: str,
    replication: int,
    seed: int,
) -> Step1Result:
    """Exécute une simulation minimale."""
    base_sent = 100 + int(density * 50)
    base_pdr = 0.9 - 0.25 * density
    if snir_mode.lower() in {"off", "disabled", "snir_off"}:
        base_pdr -= 0.05
    if algo.lower() in {"adaptive", "enhanced"}:
        base_pdr += 0.05
    variability = random.uniform(-0.02, 0.02)
    pdr = min(max(base_pdr + variability, 0.05), 0.99)
    received = max(0, min(base_sent, int(round(base_sent * pdr))))
    _ = (replication, seed)
    return Step1Result(sent=base_sent, received=received)
