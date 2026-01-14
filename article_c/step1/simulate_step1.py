"""Simulation de l'étape 1 (placeholder)."""

from dataclasses import dataclass

from article_c.common.metrics import packet_delivery_ratio


@dataclass
class Step1Result:
    sent: int
    received: int

    @property
    def pdr(self) -> float:
        return packet_delivery_ratio(self.received, self.sent)


def run_simulation(sent: int = 100, received: int = 92) -> Step1Result:
    """Exécute une simulation minimale."""
    return Step1Result(sent=sent, received=received)
