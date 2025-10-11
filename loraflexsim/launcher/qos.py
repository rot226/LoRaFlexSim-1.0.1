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

    SPEED_OF_LIGHT = 299_792_458.0  # m/s

    def __init__(self) -> None:
        self.active_algorithm: str | None = None
        self.clusters: list[Cluster] = []
        self.sf_limits: dict[int, dict[int, float]] = {}
        self.node_sf_access: dict[int, list[int]] = {}
        self.node_clusters: dict[int, int] = {}
        self.cluster_d_matrix: dict[int, dict[int, int]] = {}

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
        if self.clusters:
            self._update_qos_context(simulator)
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

    # --- Calculs QoS -------------------------------------------------------
    def _update_qos_context(self, simulator) -> None:
        if not self.clusters:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self.cluster_d_matrix = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            setattr(simulator, "qos_d_matrix", {})
            return

        channel = self._reference_channel(simulator)
        if channel is None:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self.cluster_d_matrix = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            setattr(simulator, "qos_d_matrix", {})
            return

        noise_power_w = self._noise_power_w(channel)
        if noise_power_w <= 0.0:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self.cluster_d_matrix = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            setattr(simulator, "qos_d_matrix", {})
            return

        snr_requirements = self._snr_table(simulator)
        if not snr_requirements:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self.cluster_d_matrix = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            setattr(simulator, "qos_d_matrix", {})
            return

        alpha = getattr(channel, "path_loss_exp", 2.0)
        if alpha <= 0.0:
            alpha = 2.0
        frequency = getattr(channel, "frequency_hz", 868e6)
        if frequency <= 0.0:
            frequency = 868e6
        wavelength = self.SPEED_OF_LIGHT / frequency
        factor = wavelength / (4.0 * math.pi)

        nodes = list(getattr(simulator, "nodes", []))
        reference_tx_dbm = self._reference_tx_power_dbm(simulator, nodes)
        reference_tx_w = self._dbm_to_w(reference_tx_dbm)

        cluster_limits: dict[int, dict[int, float]] = {}
        base_logs = {}
        for cluster in self.clusters:
            base = self._pdr_log_term(cluster.pdr_target)
            base_logs[cluster.cluster_id] = base
            cluster_limits[cluster.cluster_id] = {
                sf: self._compute_limit(base, reference_tx_w, noise_power_w, q, alpha, factor)
                for sf, q in snr_requirements.items()
            }

        self.sf_limits = cluster_limits

        assignments = self._assign_nodes_to_clusters(nodes)
        node_sf_access: dict[int, list[int]] = {}
        node_cluster_ids: dict[int, int] = {}
        sfs = sorted(snr_requirements)
        for node, cluster in assignments.items():
            node_cluster_ids[getattr(node, "id", id(node))] = cluster.cluster_id
            setattr(node, "qos_cluster_id", cluster.cluster_id)
            base = base_logs.get(cluster.cluster_id, 0.0)
            node_tx_w = self._dbm_to_w(getattr(node, "tx_power", reference_tx_dbm))
            distance = self._nearest_gateway_distance(node, getattr(simulator, "gateways", []))
            accessible: list[int] = []
            if base > 0.0 and node_tx_w > 0.0:
                for sf in sfs:
                    limit = self._compute_limit(
                        base,
                        node_tx_w,
                        noise_power_w,
                        snr_requirements[sf],
                        alpha,
                        factor,
                    )
                    if limit <= 0.0:
                        continue
                    if distance <= limit:
                        accessible.append(sf)
            node_sf_access[getattr(node, "id", id(node))] = accessible
            setattr(node, "qos_accessible_sf", accessible)
            if accessible:
                setattr(node, "qos_min_sf", accessible[0])
            else:
                setattr(node, "qos_min_sf", None)

        self.node_sf_access = node_sf_access
        self.node_clusters = node_cluster_ids
        d_matrix = self._build_d_matrix(assignments, node_sf_access, sfs)
        self.cluster_d_matrix = d_matrix
        setattr(simulator, "qos_sf_limits", cluster_limits)
        setattr(simulator, "qos_node_sf_access", node_sf_access)
        setattr(simulator, "qos_node_clusters", node_cluster_ids)
        setattr(simulator, "qos_d_matrix", d_matrix)

    @staticmethod
    def _reference_channel(simulator):
        channel = getattr(simulator, "channel", None)
        if channel is not None:
            return channel
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is None:
            return None
        channels = getattr(multichannel, "channels", None)
        if not channels:
            return None
        return channels[0]

    @staticmethod
    def _noise_power_w(channel) -> float:
        bandwidth = getattr(channel, "bandwidth", 125000.0)
        filter_bw = getattr(channel, "frontend_filter_bw", bandwidth)
        effective_bw = min(bandwidth, filter_bw)
        if effective_bw <= 0.0:
            return 0.0
        base_noise_dbm = getattr(channel, "receiver_noise_floor_dBm", -174.0)
        noise_figure = getattr(channel, "noise_figure_dB", 6.0)
        noise_dbm = base_noise_dbm + 10.0 * math.log10(effective_bw) + noise_figure
        return QoSManager._dbm_to_w(noise_dbm)

    @staticmethod
    def _snr_table(simulator) -> dict[int, float]:
        required = getattr(simulator, "REQUIRED_SNR", None)
        if not required:
            return {}
        return {sf: 10.0 ** (snr_db / 10.0) for sf, snr_db in required.items()}

    @staticmethod
    def _reference_tx_power_dbm(simulator, nodes: list) -> float:
        fixed = getattr(simulator, "fixed_tx_power", None)
        if fixed is not None:
            return float(fixed)
        if nodes:
            return float(getattr(nodes[0], "tx_power", 14.0))
        return 14.0

    @staticmethod
    def _dbm_to_w(power_dbm: float) -> float:
        return 0.0 if power_dbm is None else 10.0 ** ((power_dbm - 30.0) / 10.0)

    @staticmethod
    def _pdr_log_term(pdr: float) -> float:
        if pdr is None or not (0.0 < pdr < 1.0):
            return 0.0
        return -math.log(pdr)

    @staticmethod
    def _compute_limit(
        base_log: float,
        tx_power_w: float,
        noise_power_w: float,
        q_value: float,
        alpha: float,
        factor: float,
    ) -> float:
        if base_log <= 0.0 or tx_power_w <= 0.0 or noise_power_w <= 0.0 or q_value <= 0.0:
            return 0.0
        ratio = base_log * tx_power_w / (noise_power_w * q_value)
        if ratio <= 0.0:
            return 0.0
        return (ratio ** (1.0 / alpha)) * factor

    def _assign_nodes_to_clusters(self, nodes: list) -> dict[object, Cluster]:
        if not nodes or not self.clusters:
            return {}
        sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "id", id(n)))
        total = len(sorted_nodes)
        quotas: list[int] = []
        fractions: list[tuple[float, int]] = []
        assigned = 0
        for index, cluster in enumerate(self.clusters):
            expected = cluster.device_share * total
            base = math.floor(expected)
            quotas.append(base)
            fractions.append((expected - base, index))
            assigned += base
        remaining = total - assigned
        if remaining > 0:
            fractions.sort(key=lambda item: (item[0], -self.clusters[item[1]].cluster_id), reverse=True)
            for _, idx in fractions:
                if remaining <= 0:
                    break
                quotas[idx] += 1
                remaining -= 1
        elif remaining < 0:
            fractions.sort(key=lambda item: (item[0], self.clusters[item[1]].cluster_id))
            for _, idx in fractions:
                if remaining >= 0:
                    break
                if quotas[idx] > 0:
                    quotas[idx] -= 1
                    remaining += 1

        assignments: dict[object, Cluster] = {}
        node_index = 0
        for quota, cluster in zip(quotas, self.clusters):
            for _ in range(quota):
                if node_index >= total:
                    break
                node = sorted_nodes[node_index]
                assignments[node] = cluster
                node_index += 1
        while node_index < total:
            node = sorted_nodes[node_index]
            assignments[node] = self.clusters[-1]
            node_index += 1
        return assignments

    def _build_d_matrix(
        self,
        assignments: dict[object, Cluster],
        node_sf_access: dict[int, list[int]],
        sfs: Sequence[int],
    ) -> dict[int, dict[int, int]]:
        if not assignments:
            return {}

        matrix: dict[int, dict[int, int]] = {
            cluster.cluster_id: {sf: 0 for sf in sfs}
            for cluster in self.clusters
        }

        for node, cluster in assignments.items():
            node_id = getattr(node, "id", id(node))
            accessible = node_sf_access.get(node_id) or []
            if not accessible:
                continue
            min_sf = accessible[0]
            if min_sf in matrix[cluster.cluster_id]:
                matrix[cluster.cluster_id][min_sf] += 1
        return matrix

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

