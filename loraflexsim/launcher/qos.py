"""Gestionnaires de stratégies QoS pour LoRaFlexSim.

Ce module propose une implémentation légère de deux variantes de l'algorithme
MixRA. L'objectif n'est pas de reproduire fidèlement toutes les optimisations
présentées dans la littérature, mais de fournir une base cohérente pour
expérimenter rapidement des répartitions SF/puissance orientées QoS.

Chaque algorithme s'appuie sur la distance entre un nœud et la passerelle la
plus proche afin d'attribuer un couple ``(SF, puissance)``. Les méthodes
peuvent être affinées par la suite si un modèle plus précis est nécessaire.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


# Mapping lisible pour l'interface utilisateur :
# clé (affichage) -> suffixe de méthode interne.
QOS_ALGORITHMS = {
    "MixRA-Opt": "mixra_opt",
    "MixRA-H": "mixra_h",
}


@dataclass(slots=True)
class _NodeDistance:
    """Couple ``(node, distance)`` triable par distance."""

    node: object
    distance: float


class QoSManager:
    """Applique une stratégie QoS aux nœuds du simulateur."""

    def __init__(self) -> None:
        self.active_algorithm: str | None = None

    # --- Interface publique -------------------------------------------------
    def apply(self, simulator, algorithm: str) -> None:
        """Active ``algorithm`` sur ``simulator``.

        ``algorithm`` doit correspondre à une clé de :data:`QOS_ALGORITHMS`.
        """

        if algorithm not in QOS_ALGORITHMS:
            raise ValueError(f"Algorithme QoS inconnu : {algorithm}")
        method_name = f"_apply_{QOS_ALGORITHMS[algorithm]}"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Implémentation manquante pour {algorithm}")
        if not getattr(simulator, "nodes", None):
            # Rien à faire si aucun nœud n'est présent.
            self.active_algorithm = algorithm
            setattr(simulator, "qos_algorithm", algorithm)
            setattr(simulator, "qos_active", True)
            return
        self.active_algorithm = algorithm
        method(simulator)
        setattr(simulator, "qos_algorithm", algorithm)
        setattr(simulator, "qos_active", True)

    # --- Utilitaires internes ----------------------------------------------
    @staticmethod
    def _nearest_gateway_distance(node, gateways: Iterable) -> float:
        if not gateways:
            return 0.0
        return min(
            math.hypot(node.x - gw.x, node.y - gw.y)
            for gw in gateways
        )

    def _sorted_distances(self, simulator) -> list[_NodeDistance]:
        return sorted(
            (
                _NodeDistance(
                    node=node,
                    distance=self._nearest_gateway_distance(node, simulator.gateways),
                )
                for node in simulator.nodes
            ),
            key=lambda entry: entry.distance,
        )

    @staticmethod
    def _assign_tx_power(sf_index: int) -> float:
        """Retourne une puissance TX bornée entre 2 et 20 dBm.

        Les puissances augmentent graduellement avec le SF pour mimer les
        besoins de portée tout en conservant une enveloppe réaliste.
        """

        base = 8.0
        step = 2.0
        power = base + sf_index * step
        return max(2.0, min(20.0, power))

    # --- Implémentations MixRA ---------------------------------------------
    def _apply_mixra_opt(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        sfs = [7, 8, 9, 10, 11, 12]
        chunk_size = max(1, math.ceil(len(pairs) / len(sfs)))
        for index, entry in enumerate(pairs):
            sf_index = min(index // chunk_size, len(sfs) - 1)
            sf = sfs[sf_index]
            entry.node.sf = sf
            entry.node.tx_power = self._assign_tx_power(sf_index)

    def _apply_mixra_h(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        sfs = [7, 8, 9, 10, 11, 12]
        # Répartition heuristique : favorise les petits SF tout en réservant une
        # part non négligeable aux valeurs élevées pour couvrir les zones plus
        # éloignées.
        distribution = [0.28, 0.22, 0.18, 0.14, 0.10, 0.08]
        total = len(pairs)
        cumulative = 0
        boundaries = []
        for share in distribution:
            cumulative += share * total
            boundaries.append(round(cumulative))
        boundaries[-1] = total
        current = 0
        for sf_index, boundary in enumerate(boundaries):
            sf = sfs[sf_index]
            while current < min(boundary, total):
                entry = pairs[current]
                entry.node.sf = sf
                entry.node.tx_power = self._assign_tx_power(sf_index)
                current += 1
        while current < total:
            entry = pairs[current]
            entry.node.sf = sfs[-1]
            entry.node.tx_power = self._assign_tx_power(len(sfs) - 1)
            current += 1

