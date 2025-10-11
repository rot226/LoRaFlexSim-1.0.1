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

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

try:  # pragma: no cover - SciPy est une dépendance obligatoire à l'exécution
    from scipy.special import lambertw
except Exception:  # pragma: no cover - gestion defensive
    lambertw = None


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
        self._last_distribution: dict[int, dict[int, dict[int, float]]] = {}

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
            self._last_distribution = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            return

        channel = self._reference_channel(simulator)
        if channel is None:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self._last_distribution = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            return

        noise_power_w = self._noise_power_w(channel)
        if noise_power_w <= 0.0:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self._last_distribution = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
            return

        snr_requirements = self._snr_table(simulator)
        if not snr_requirements:
            self.sf_limits = {}
            self.node_sf_access = {}
            self.node_clusters = {}
            self._last_distribution = {}
            setattr(simulator, "qos_sf_limits", {})
            setattr(simulator, "qos_node_sf_access", {})
            setattr(simulator, "qos_node_clusters", {})
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
        for node, cluster in assignments.items():
            node_cluster_ids[getattr(node, "id", id(node))] = cluster.cluster_id
            setattr(node, "qos_cluster_id", cluster.cluster_id)
            base = base_logs.get(cluster.cluster_id, 0.0)
            node_tx_w = self._dbm_to_w(getattr(node, "tx_power", reference_tx_dbm))
            distance = self._nearest_gateway_distance(node, getattr(simulator, "gateways", []))
            accessible: list[int] = []
            if base > 0.0 and node_tx_w > 0.0:
                for sf in sorted(snr_requirements):
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

        self.node_sf_access = node_sf_access
        self.node_clusters = node_cluster_ids
        setattr(simulator, "qos_sf_limits", cluster_limits)
        setattr(simulator, "qos_node_sf_access", node_sf_access)
        setattr(simulator, "qos_node_clusters", node_cluster_ids)
        self._last_distribution = self._compute_cluster_distribution(simulator)

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
    def _channel_list(simulator) -> list:
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            candidates = list(getattr(multichannel, "channels", []) or [])
            if candidates:
                return candidates
        channel = getattr(simulator, "channel", None)
        return [channel] if channel is not None else []

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

    # --- Trafic offert et capacité ---------------------------------------
    def _compute_cluster_distribution(
        self, simulator
    ) -> dict[int, dict[int, dict[int, float]]]:
        """Calcule :math:`D_{k,j,f}` (répartition des nœuds par cluster/SF/canal)."""

        nodes = list(getattr(simulator, "nodes", []) or [])
        if not nodes or not self.clusters:
            return {}

        channels = self._channel_list(simulator)
        if not channels:
            return {}
        channel_index = {id(ch): idx for idx, ch in enumerate(channels)}

        totals: defaultdict[int, int] = defaultdict(int)
        counts: defaultdict[int, defaultdict[int, defaultdict[int, int]]]
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for node in nodes:
            cluster_id = getattr(node, "qos_cluster_id", None)
            sf = getattr(node, "sf", None)
            if cluster_id is None or sf is None:
                continue
            channel = getattr(node, "channel", None)
            channel_id = channel_index.get(id(channel), 0)
            totals[cluster_id] += 1
            counts[cluster_id][int(sf)][channel_id] += 1

        distribution: dict[int, dict[int, dict[int, float]]] = {}
        for cluster_id, sf_counts in counts.items():
            total = totals.get(cluster_id, 0)
            if total <= 0:
                continue
            sf_distribution: dict[int, dict[int, float]] = {}
            for sf, channel_counts in sf_counts.items():
                channel_distribution = {
                    ch_idx: count / total
                    for ch_idx, count in channel_counts.items()
                    if count > 0
                }
                if channel_distribution:
                    sf_distribution[sf] = channel_distribution
            if sf_distribution:
                distribution[cluster_id] = sf_distribution
        return distribution

    def cluster_distribution(
        self, simulator
    ) -> dict[int, dict[int, dict[int, float]]]:
        """Retourne la dernière estimation de :math:`D_{k,j,f}`.

        Le dictionnaire retourné est structuré comme suit::

            {cluster_id: {sf: {channel_index: proportion}}}

        Les proportions sont normalisées par rapport au nombre total de nœuds
        du cluster, ce qui permet un calcul direct du trafic offert.
        """

        self._last_distribution = self._compute_cluster_distribution(simulator)
        return self._last_distribution

    def offered_traffic(
        self,
        simulator,
        *,
        payload_size: int | None = None,
    ) -> dict[int, dict[int, dict[int, float]]]:
        r"""Calcule :math:`\nu_{k,j,f}` pour chaque cluster/SF/canal."""

        if not self.clusters:
            return {}

        reference_channel = self._reference_channel(simulator)
        if reference_channel is None:
            return {}

        distribution = self.cluster_distribution(simulator)
        if not distribution:
            return {}

        if payload_size is None:
            payload_size = int(getattr(simulator, "payload_size_bytes", 20))

        tau_cache: dict[int, float] = {}
        loads: dict[int, dict[int, dict[int, float]]] = {}

        for cluster in self.clusters:
            cluster_dist = distribution.get(cluster.cluster_id)
            if not cluster_dist:
                continue
            sf_loads: dict[int, dict[int, float]] = {}
            for sf, channel_dist in cluster_dist.items():
                tau = tau_cache.setdefault(
                    sf, reference_channel.airtime(sf, payload_size=payload_size)
                )
                offered = {
                    ch_idx: channel_share * cluster.arrival_rate * tau
                    for ch_idx, channel_share in channel_dist.items()
                    if channel_share > 0.0
                }
                if offered:
                    sf_loads[sf] = offered
            if sf_loads:
                loads[cluster.cluster_id] = sf_loads
        return loads

    @staticmethod
    def approximate_capacity(pdr_target: float, delta: float = 0.0) -> float:
        r"""Approximation analytique de :math:`\nu_k^{\max}` via Lambert W."""

        if not (0.0 < pdr_target <= 1.0):
            raise ValueError("La cible PDR doit appartenir à ]0, 1]")
        if lambertw is None:
            raise RuntimeError(
                "SciPy est requis pour calculer la fonction de Lambert W"
            )
        xi = float(delta) + 1.0
        argument = -xi * math.exp(xi) / pdr_target
        result = lambertw(argument, -1)
        value = result.real if isinstance(result, complex) else float(result)
        return max(0.0, -0.5 * value - 0.5 * xi)

    @staticmethod
    def refine_capacity(
        approx: float,
        pdr_target: float,
        pdr_function: Callable[[float], float] | None,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> float:
        r"""Affinement numérique de :math:`\nu_k^{\max}` à partir d'un modèle ``h``."""

        if pdr_function is None or approx < 0.0:
            return max(0.0, approx)

        def evaluate(load: float) -> float:
            value = pdr_function(load)
            if value is None or math.isnan(value):
                raise ValueError("La fonction PDR doit retourner une valeur réelle")
            return float(value)

        target = float(pdr_target)
        if not (0.0 < target <= 1.0):
            raise ValueError("La cible PDR doit appartenir à ]0, 1]")

        try:
            approx_value = evaluate(approx)
        except ValueError:
            return max(0.0, approx)

        lower = 0.0
        upper = max(approx, 1e-9)

        try:
            lower_value = evaluate(lower)
        except ValueError:
            lower_value = 1.0

        if lower_value < target:
            return 0.0

        if approx_value > target:
            lower = approx
            lower_value = approx_value
            upper = max(approx * 2.0, approx + 1e-6)
            it = 0
            while True:
                try:
                    upper_value = evaluate(upper)
                except ValueError:
                    upper_value = 0.0
                if upper_value <= target or it >= max_iter:
                    break
                lower = upper
                lower_value = upper_value
                upper *= 2.0
                it += 1
        else:
            upper = approx
            upper_value = approx_value
            it = 0
            while True:
                mid = upper * 0.5
                if mid <= 1e-12 or it >= max_iter:
                    break
                try:
                    mid_value = evaluate(mid)
                except ValueError:
                    mid_value = 0.0
                if mid_value >= target:
                    upper = mid
                    upper_value = mid_value
                else:
                    break
                it += 1

        try:
            upper_value
        except UnboundLocalError:
            upper_value = evaluate(upper)

        if upper_value > target:
            return max(0.0, upper)

        for _ in range(max_iter):
            midpoint = 0.5 * (lower + upper)
            try:
                value = evaluate(midpoint)
            except ValueError:
                value = 0.0
            if abs(value - target) <= tol:
                return max(0.0, midpoint)
            if value > target:
                lower = midpoint
            else:
                upper = midpoint
        return max(0.0, 0.5 * (lower + upper))

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

