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
from typing import Iterable, Sequence


# Mapping lisible pour l'interface utilisateur :
# clé (affichage) -> suffixe de méthode interne.
QOS_ALGORITHMS = {
    "MixRA-Opt": "mixra_opt",
    "MixRA-H": "mixra_h",
}

__all__ = ["Cluster", "build_clusters", "QoSManager"]


@dataclass(frozen=True, slots=True)
class Cluster:
    r"""Représentation d'un cluster QoS défini par l'utilisateur.

    Chaque cluster regroupe une proportion d'équipements possédant un même
    profil de trafic (:math:`\lambda_k`) et une cible de fiabilité (``PDR``).
    """

    cluster_id: int
    device_share: float
    arrival_rate: float
    pdr_target: float


def build_clusters(
    cluster_count: int,
    *,
    proportions: Sequence[float],
    arrival_rates: Sequence[float],
    pdr_targets: Sequence[float],
    id_start: int = 1,
) -> list[Cluster]:
    r"""Crée ``cluster_count`` instances :class:`Cluster` à partir des paramètres.

    Args:
        cluster_count: nombre de clusters à instancier.
        proportions: proportions d'équipements par cluster (somme ≈ 1).
        arrival_rates: taux d'arrivée :math:`\lambda_k` correspondants.
        pdr_targets: objectifs de fiabilité ``PDR`` par cluster.
        id_start: valeur de départ pour l'identifiant (par défaut ``1``).

    Returns:
        Une liste de :class:`Cluster` validée et normalisée.

    Raises:
        ValueError: si les longueurs ne correspondent pas, si une valeur est
            hors borne ou si la somme des proportions diffère significativement
            de 1.
    """

    if cluster_count <= 0:
        raise ValueError("Le nombre de clusters doit être strictement positif")
    lengths = (len(proportions), len(arrival_rates), len(pdr_targets))
    if any(length != cluster_count for length in lengths):
        raise ValueError(
            "Les longueurs des proportions, taux d'arrivée et cibles PDR doivent "
            "correspondre au nombre de clusters"
        )
    if any(share <= 0.0 for share in proportions):
        raise ValueError("Chaque proportion doit être strictement positive")
    total_share = float(sum(proportions))
    if not math.isclose(total_share, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("La somme des proportions doit être égale à 1")
    if any(rate <= 0.0 for rate in arrival_rates):
        raise ValueError("Chaque taux d'arrivée doit être strictement positif")
    if any(not (0.0 < target <= 1.0) for target in pdr_targets):
        raise ValueError("Chaque cible PDR doit appartenir à l'intervalle ]0, 1]")

    clusters: list[Cluster] = []
    for index in range(cluster_count):
        clusters.append(
            Cluster(
                cluster_id=id_start + index,
                device_share=float(proportions[index]),
                arrival_rate=float(arrival_rates[index]),
                pdr_target=float(pdr_targets[index]),
            )
        )
    return clusters


@dataclass(slots=True)
class _NodeDistance:
    """Couple ``(node, distance)`` triable par distance."""

    node: object
    distance: float


class QoSManager:
    """Applique une stratégie QoS aux nœuds du simulateur."""

    def __init__(self) -> None:
        self.active_algorithm: str | None = None
        self.clusters: list[Cluster] = []

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

    def configure_clusters(
        self,
        cluster_count: int,
        *,
        proportions: Sequence[float],
        arrival_rates: Sequence[float],
        pdr_targets: Sequence[float],
    ) -> list[Cluster]:
        """Initialise ``cluster_count`` clusters à partir des paramètres fournis."""

        self.clusters = build_clusters(
            cluster_count,
            proportions=proportions,
            arrival_rates=arrival_rates,
            pdr_targets=pdr_targets,
        )
        return list(self.clusters)

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

