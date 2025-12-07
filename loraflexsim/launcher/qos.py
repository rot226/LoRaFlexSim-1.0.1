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

import importlib.util
from importlib import import_module
from collections import deque, defaultdict
from dataclasses import dataclass
import math
from types import ModuleType
from typing import Iterable, Sequence

try:  # pragma: no cover - gestion de l'import partiel lors des tests unitaires
    from . import ADR_MODULES, adr_max, adr_standard_1
except ImportError:  # pragma: no cover - fallback pour les imports partiels
    ADR_MODULES = {}
    adr_max = None
    adr_standard_1 = None

_SCIPY_SPEC = importlib.util.find_spec("scipy.optimize")
if _SCIPY_SPEC is not None:
    from scipy.optimize import minimize  # type: ignore[assignment]
else:  # pragma: no cover - dépendance optionnelle
    minimize = None  # type: ignore[assignment]


# Mapping lisible pour l'interface utilisateur :
# clé (affichage) -> suffixe de méthode interne.
QOS_ALGORITHMS = {
    "ADR-Pure": "adr_pure",
    "APRA-like": "apra_like",
    "Aimi-like": "aimi_like",
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
        self.sf_airtimes: dict[int, float] = {}
        self.cluster_offered_traffic: dict[int, dict[int, dict[int, float]]] = {}
        self.cluster_offered_totals: dict[int, float] = {}
        self.cluster_capacity_limits: dict[int, float] = {}
        self.cluster_interference: dict[int, float] = {}
        self.cluster_sf_channel_capacity: dict[int, dict[int, dict[int, float]]] = {}
        self.capacity_margin: float = 0.1
        self.reconfig_interval_s: float = 60.0
        self.pdr_drift_threshold: float = 0.1
        self.traffic_drift_threshold: float = 0.25
        self.max_clusters_per_channel: int | None = None
        self.max_clusters_per_min_sf: int | None = None
        self._last_reconfig_time: float | None = None
        self._last_node_ids: set[int] = set()
        self._last_recent_pdr: dict[int, float] = {}
        self._last_arrival_rates: dict[int, float] = {}
        self._last_assignments: dict[int, tuple[int, int]] = {}
        self.out_of_service_queue: deque[tuple[int, str]] = deque()

    # --- Interface publique -------------------------------------------------
    def apply(
        self,
        simulator,
        algorithm: str,
        *,
        use_snir: bool | None = None,
        inter_sf_coupling: float | None = None,
        capture_thresholds: Sequence[float] | None = None,
    ) -> None:
        """Active ``algorithm`` sur ``simulator``.

        ``algorithm`` doit correspondre à une clé de :data:`QOS_ALGORITHMS`.
        Les paramètres optionnels permettent d'ajuster le modèle radio (SNIR,
        couplage inter-SF, seuils de capture) sans modifier le reste du
        simulateur.
        """

        if algorithm not in QOS_ALGORITHMS:
            raise ValueError(f"Algorithme QoS inconnu : {algorithm}")
        method_name = f"_apply_{QOS_ALGORITHMS[algorithm]}"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Implémentation manquante pour {algorithm}")
        shared_queue = getattr(simulator, "out_of_service_queue", None)
        if shared_queue is not None:
            self.out_of_service_queue = shared_queue
        if self.clusters and self._should_refresh_context(simulator):
            self._update_qos_context(simulator)
        self._configure_radio_model(
            simulator,
            use_snir=use_snir,
            inter_sf_coupling=inter_sf_coupling,
            capture_thresholds=capture_thresholds,
        )
        setattr(simulator, "qos_manager", self)
        if not getattr(simulator, "nodes", None):
            # Rien à faire si aucun nœud n'est présent.
            self.active_algorithm = algorithm
            setattr(simulator, "qos_algorithm", algorithm)
            setattr(simulator, "qos_active", True)
            self._broadcast_control_updates(simulator)
            hook = getattr(simulator, "_on_qos_applied", None)
            if callable(hook):
                hook(self)
            return
        self.active_algorithm = algorithm
        method(simulator)
        setattr(simulator, "qos_algorithm", algorithm)
        setattr(simulator, "qos_active", True)
        self._broadcast_control_updates(simulator)
        hook = getattr(simulator, "_on_qos_applied", None)
        if callable(hook):
            hook(self)

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
        self._last_reconfig_time = None
        self._last_node_ids = set()
        self._last_recent_pdr = {}
        self._last_assignments = {}
        return list(self.clusters)

    def set_mixra_cluster_limits(
        self,
        *,
        channel_cluster_limit: int | None = None,
        sf_cluster_limit: int | None = None,
    ) -> None:
        """Définit les bornes MixRA ``D`` (canaux) et ``F`` (SF minimaux)."""

        if channel_cluster_limit is not None:
            if channel_cluster_limit < 0:
                raise ValueError("La borne D doit être positive ou nulle.")
            self.max_clusters_per_channel = int(channel_cluster_limit)
        else:
            self.max_clusters_per_channel = None

        if sf_cluster_limit is not None:
            if sf_cluster_limit < 0:
                raise ValueError("La borne F doit être positive ou nulle.")
            self.max_clusters_per_min_sf = int(sf_cluster_limit)
        else:
            self.max_clusters_per_min_sf = None

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

    def _broadcast_control_updates(self, simulator) -> None:
        server = getattr(simulator, "network_server", None)
        nodes = getattr(simulator, "nodes", None)
        if server is None or not nodes:
            self._last_assignments = {}
            return
        updates: list[tuple[object, int, int]] = []
        for node in nodes:
            node_id = getattr(node, "id", None)
            if node_id is None:
                continue
            sf_value = int(getattr(node, "sf", 0))
            channel_obj = getattr(node, "channel", None)
            channel_index = simulator.channel_index(channel_obj)
            previous = self._last_assignments.get(node_id)
            state = (sf_value, channel_index)
            if previous != state:
                updates.append((node, sf_value, channel_index))
            self._last_assignments[node_id] = state
            node.assigned_channel_index = channel_index
        if not updates:
            return
        control_channel = getattr(simulator, "control_channel", None)
        current_time = getattr(simulator, "current_time", None)
        from .lorawan import ControlUpdate

        for node, sf_value, channel_index in updates:
            payload = ControlUpdate(sf_value, channel_index).to_bytes()
            server.send_downlink(
                node,
                payload=payload,
                at_time=current_time,
                channel=control_channel,
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

    @staticmethod
    def _normalize_capture_thresholds(
        capture_thresholds: Sequence[float] | None,
    ) -> float | dict[int, float] | None:
        if capture_thresholds is None:
            return None
        values = [float(value) for value in capture_thresholds if value is not None]
        if not values:
            return None
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Les seuils de capture doivent être finis.")
        if len(values) == 1:
            return values[0]
        return {sf: values[idx] for idx, sf in enumerate(range(7, 7 + len(values)))}

    def _configure_radio_model(
        self,
        simulator,
        *,
        use_snir: bool | None,
        inter_sf_coupling: float | None,
        capture_thresholds: Sequence[float] | None,
    ) -> None:
        if simulator is None:
            return
        channels: list[object] = []
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channels.append(base_channel)
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            channels.extend(getattr(multichannel, "channels", []) or [])
        if not channels:
            return

        threshold_value = self._normalize_capture_thresholds(capture_thresholds)
        for channel in channels:
            if use_snir is not None:
                channel.use_snir = bool(use_snir)
            if inter_sf_coupling is not None:
                coupling = float(inter_sf_coupling)
                channel.alpha_isf = coupling
                channel.orthogonal_sf = coupling <= 0.0
            if threshold_value is not None:
                if isinstance(threshold_value, dict) and str(getattr(channel, "phy_model", "")).startswith("omnet"):
                    channel.capture_threshold_dB = float(next(iter(threshold_value.values())))
                else:
                    channel.capture_threshold_dB = threshold_value

    # --- Calculs QoS -------------------------------------------------------
    def _update_qos_context(self, simulator) -> None:
        if not self.clusters:
            self._clear_qos_state(simulator)
            return

        channel = self._reference_channel(simulator)
        if channel is None:
            self._clear_qos_state(simulator)
            return

        channel_map: dict[int, object | None] = {}
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            for index, ch in enumerate(getattr(multichannel, "channels", []) or []):
                channel_index = getattr(ch, "channel_index", index)
                channel_map[channel_index] = ch
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            base_index = getattr(base_channel, "channel_index", 0)
            channel_map.setdefault(base_index, base_channel)
        if not channel_map:
            channel_map[0] = channel
        channel_indices = sorted(channel_map)
        if not channel_indices:
            channel_indices = [0]

        noise_power_w = self._noise_power_w(channel)
        if noise_power_w <= 0.0:
            self._clear_qos_state(simulator)
            return

        snr_requirements = self._snr_table(simulator)
        if not snr_requirements:
            self._clear_qos_state(simulator)
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
        airtimes = self._compute_sf_airtimes(simulator, sfs)
        offered = self._compute_offered_traffic(assignments, node_sf_access, airtimes)
        totals = {
            cluster_id: sum(sum(ch.values()) for ch in sf_map.values())
            for cluster_id, sf_map in offered.items()
        }
        total_load = sum(totals.values())
        interference = {
            cluster_id: max(total_load - totals.get(cluster_id, 0.0), 0.0)
            for cluster_id in totals
        }
        margin_factor = max(0.0, 1.0 - max(self.capacity_margin, 0.0))
        capacities = {}
        for cluster in self.clusters:
            capacity = self._capacity_from_pdr(
                cluster.pdr_target,
                interference.get(cluster.cluster_id, 0.0),
            )
            if capacity > 0.0 and margin_factor > 0.0:
                capacity *= margin_factor
            elif margin_factor <= 0.0:
                capacity = 0.0
            capacities[cluster.cluster_id] = capacity

        channel_totals: dict[int, float] = {}
        for cluster_map in offered.values():
            for sf_map in cluster_map.values():
                for channel_index, value in sf_map.items():
                    channel_totals[channel_index] = channel_totals.get(channel_index, 0.0) + value

        sf_channel_capacity: dict[int, dict[int, dict[int, float]]] = {}
        for cluster in self.clusters:
            cluster_id = cluster.cluster_id
            per_sf: dict[int, dict[int, float]] = {}
            cluster_offered = offered.get(cluster_id, {})
            for sf in sfs:
                per_channel: dict[int, float] = {}
                sf_offered = cluster_offered.get(sf, {})
                for channel_index in channel_indices:
                    other_load = channel_totals.get(channel_index, 0.0) - sf_offered.get(channel_index, 0.0)
                    if other_load < 0.0:
                        other_load = 0.0
                    limit = self._capacity_from_pdr(cluster.pdr_target, other_load)
                    if limit > 0.0 and margin_factor > 0.0:
                        limit *= margin_factor
                    elif margin_factor <= 0.0:
                        limit = 0.0
                    per_channel[channel_index] = float(limit)
                if per_channel:
                    per_sf[sf] = per_channel
            if per_sf:
                sf_channel_capacity[cluster_id] = per_sf
        self.sf_airtimes = airtimes
        self.cluster_offered_traffic = offered
        self.cluster_offered_totals = totals
        self.cluster_interference = interference
        self.cluster_capacity_limits = capacities
        self.cluster_sf_channel_capacity = sf_channel_capacity
        cluster_config = {
            cluster.cluster_id: {
                "arrival_rate": float(cluster.arrival_rate),
                "pdr_target": float(cluster.pdr_target),
                "device_share": float(cluster.device_share),
            }
            for cluster in self.clusters
        }
        setattr(simulator, "qos_sf_limits", cluster_limits)
        setattr(simulator, "qos_node_sf_access", node_sf_access)
        setattr(simulator, "qos_node_clusters", node_cluster_ids)
        setattr(simulator, "qos_d_matrix", d_matrix)
        setattr(simulator, "qos_sf_airtimes", airtimes)
        setattr(simulator, "qos_offered_traffic", offered)
        setattr(simulator, "qos_offered_totals", totals)
        setattr(simulator, "qos_interference", interference)
        setattr(simulator, "qos_capacity_limits", capacities)
        setattr(simulator, "qos_sf_channel_capacity", sf_channel_capacity)
        setattr(simulator, "qos_clusters_config", cluster_config)

        node_ids = {getattr(node, "id", id(node)) for node in nodes}
        self._last_node_ids = node_ids
        self._last_recent_pdr = {
            getattr(node, "id", id(node)): float(getattr(node, "recent_pdr", 0.0) or 0.0)
            for node in nodes
        }
        self._last_arrival_rates = {}
        for node in nodes:
            node_id = getattr(node, "id", id(node))
            rate = self._node_arrival_rate(node)
            if rate is not None:
                self._last_arrival_rates[node_id] = rate
        current_time = getattr(simulator, "current_time", None)
        try:
            time_value = float(current_time)
        except (TypeError, ValueError):
            time_value = None
        if time_value is not None and math.isfinite(time_value):
            self._last_reconfig_time = time_value

    def _clear_qos_state(self, simulator) -> None:
        self.sf_limits = {}
        self.node_sf_access = {}
        self.node_clusters = {}
        self.cluster_d_matrix = {}
        self.sf_airtimes = {}
        self.cluster_offered_traffic = {}
        self.cluster_offered_totals = {}
        self.cluster_capacity_limits = {}
        self.cluster_interference = {}
        self.cluster_sf_channel_capacity = {}
        setattr(simulator, "qos_sf_limits", {})
        setattr(simulator, "qos_node_sf_access", {})
        setattr(simulator, "qos_node_clusters", {})
        setattr(simulator, "qos_d_matrix", {})
        setattr(simulator, "qos_sf_airtimes", {})
        setattr(simulator, "qos_offered_traffic", {})
        setattr(simulator, "qos_offered_totals", {})
        setattr(simulator, "qos_capacity_limits", {})
        setattr(simulator, "qos_interference", {})
        setattr(simulator, "qos_sf_channel_capacity", {})
        setattr(simulator, "qos_clusters_config", {})
        self._last_node_ids = set()
        self._last_recent_pdr = {}
        self._last_arrival_rates = {}
        self._last_reconfig_time = None

    def _should_refresh_context(self, simulator) -> bool:
        if not self.clusters:
            return False
        if not self.node_sf_access:
            return True
        nodes = list(getattr(simulator, "nodes", []) or [])
        node_ids = {getattr(node, "id", id(node)) for node in nodes}
        if node_ids != self._last_node_ids:
            return True
        if self._last_reconfig_time is None:
            return True
        if self.pdr_drift_threshold > 0.0 and nodes:
            for node in nodes:
                node_id = getattr(node, "id", id(node))
                previous = self._last_recent_pdr.get(node_id)
                if previous is None:
                    continue
                try:
                    current = float(getattr(node, "recent_pdr", None))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(current):
                    continue
                if abs(current - previous) >= self.pdr_drift_threshold:
                    return True
        traffic_threshold = self.traffic_drift_threshold
        if traffic_threshold > 0.0 and nodes:
            for node in nodes:
                node_id = getattr(node, "id", id(node))
                previous_rate = self._last_arrival_rates.get(node_id)
                if previous_rate is None:
                    continue
                if previous_rate <= 0.0:
                    continue
                current_rate = self._node_arrival_rate(node)
                if current_rate is None:
                    continue
                if current_rate <= 0.0:
                    return True
                if abs(current_rate - previous_rate) / previous_rate >= traffic_threshold:
                    return True
        interval = self.reconfig_interval_s
        if interval is None or interval <= 0.0:
            return False
        current_time = getattr(simulator, "current_time", None)
        try:
            time_value = float(current_time)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(time_value):
            return False
        if time_value < self._last_reconfig_time:
            return True
        return time_value - self._last_reconfig_time >= interval

    @staticmethod
    def _node_arrival_rate(node) -> float | None:
        """Calcule le taux d'arrivée empirique associé à ``node``.

        Le calcul s'appuie sur les compteurs ``arrival_interval_sum`` et
        ``arrival_interval_count`` mis à jour par le simulateur. La valeur
        retournée est ``None`` si les informations sont indisponibles ou
        incohérentes.
        """

        interval_sum = getattr(node, "arrival_interval_sum", None)
        interval_count = getattr(node, "arrival_interval_count", None)
        try:
            sum_value = float(interval_sum)
            count_value = float(interval_count)
        except (TypeError, ValueError):
            return None
        if count_value <= 0.0 or sum_value <= 0.0:
            return None
        rate = count_value / sum_value
        return rate if math.isfinite(rate) and rate > 0.0 else None

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
        if (
            base_log <= 0.0
            or tx_power_w <= 0.0
            or noise_power_w <= 0.0
            or q_value <= 0.0
            or alpha <= 0.0
            or factor <= 0.0
        ):
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

    def _compute_sf_airtimes(self, simulator, sfs: Sequence[int]) -> dict[int, float]:
        channel = self._reference_channel(simulator)
        payload = getattr(simulator, "payload_size_bytes", 20)
        airtimes: dict[int, float] = {}
        if channel is None:
            return {sf: 0.0 for sf in sfs}
        for sf in sfs:
            try:
                airtimes[sf] = float(channel.airtime(sf, payload_size=payload))
            except Exception:
                airtimes[sf] = 0.0
        return airtimes

    def _compute_offered_traffic(
        self,
        assignments: dict[object, Cluster],
        node_sf_access: dict[int, list[int]],
        airtimes: dict[int, float],
    ) -> dict[int, dict[int, dict[int, float]]]:
        offered: dict[int, dict[int, dict[int, float]]] = {
            cluster.cluster_id: {sf: {} for sf in airtimes}
            for cluster in self.clusters
        }
        for node, cluster in assignments.items():
            node_id = getattr(node, "id", id(node))
            accessible = node_sf_access.get(node_id) or []
            if not accessible:
                continue
            sf = accessible[0]
            tau = airtimes.get(sf, 0.0)
            if tau <= 0.0:
                continue
            channel_index = 0
            node_channel = getattr(node, "channel", None)
            if node_channel is not None:
                channel_index = getattr(node_channel, "channel_index", 0)
            cluster_map = offered.setdefault(cluster.cluster_id, {})
            sf_map = cluster_map.setdefault(sf, {})
            sf_map[channel_index] = sf_map.get(channel_index, 0.0) + cluster.arrival_rate * tau
        return offered

    @staticmethod
    def _capacity_from_pdr(pdr: float, delta: float = 0.0) -> float:
        if pdr is None or pdr <= 0.0:
            return 0.0
        xi = max(delta + 1.0, 1e-9)
        argument = -xi * math.exp(-xi) * pdr
        if argument < -1.0 / math.e:
            argument = -1.0 / math.e
        if argument >= 0.0:
            return 0.0
        value = QoSManager._lambertw_neg1(argument)
        nu = -0.5 * value - xi / 2.0
        if not math.isfinite(nu) or nu < 0.0:
            return 0.0
        return nu

    @staticmethod
    def _lambertw_neg1(x: float) -> float:
        if x >= 0.0:
            raise ValueError("Lambert W_{-1} indéfini pour x >= 0")
        limit = -1.0 / math.e
        if x < limit:
            x = limit
        if math.isclose(x, limit, rel_tol=0.0, abs_tol=1e-15):
            return -1.0
        if x > -0.1:
            w = -1.0
        else:
            w = math.log(-x)
        for _ in range(100):
            ew = math.exp(w)
            f = w * ew - x
            denom = ew * (w + 1.0)
            if denom == 0.0:
                break
            step = f / denom
            w_next = w - step
            if w_next > -1.0:
                w_next = (w - 1.0) / 2.0
            if abs(w_next - w) <= 1e-12 * max(1.0, abs(w_next)):
                return w_next
            w = w_next
        return w

    # --- Lignes de base ADR/APRA/AIMI -------------------------------------
    def _apply_adr_pure(self, simulator) -> None:
        self._clear_qos_state(simulator)
        setattr(simulator, "qos_active", False)
        setattr(simulator, "qos_algorithm", None)

        candidates: list[ModuleType] = []
        seen_names: set[str] = set()

        def add_candidate(module: ModuleType | None) -> None:
            if module is None:
                return
            name = getattr(module, "__name__", None)
            if not name or name in seen_names:
                return
            seen_names.add(name)
            candidates.append(module)

        # Import direct par le package courant pour éviter les ImportError
        add_candidate(adr_standard_1)
        add_candidate(adr_max)

        # Modules référencés dans ADR_MODULES (déjà importés au niveau package)
        add_candidate(ADR_MODULES.get("ADR 1"))
        add_candidate(ADR_MODULES.get("ADR-Max"))

        # En dernier recours, tentative d'import fully-qualified
        fallback_names = (
            "loraflexsim.launcher.adr_standard_1",
            "loraflexsim.launcher.adr_max",
        )

        last_error: Exception | None = None
        for module_name in fallback_names:
            if module_name in seen_names:
                continue
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            add_candidate(module)

        for module in candidates:
            apply_fn = getattr(module, "apply", None)
            if callable(apply_fn):
                try:
                    apply_fn(simulator)
                except Exception as exc:  # pragma: no cover - délégation externe
                    last_error = exc
                    continue
                return

        if last_error is not None:
            raise RuntimeError("Échec de l'initialisation ADR pour ADR-Pure") from last_error
        raise RuntimeError(
            "Aucun module ADR compatible (adr_standard_1/adr_max) n'est disponible pour ADR-Pure"
        )

    def _apply_apra_like(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        if not self.clusters or not self.node_sf_access:
            self._assign_distance_based(pairs)
            return

        channel_map: dict[int, object | None] = {}
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            for index, channel in enumerate(getattr(multichannel, "channels", []) or []):
                channel_index = getattr(channel, "channel_index", index)
                channel_map[channel_index] = channel
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channel_index = getattr(base_channel, "channel_index", 0)
            channel_map.setdefault(channel_index, base_channel)
        if not channel_map:
            channel_map[0] = base_channel
        channel_indices = sorted(channel_map)
        if not channel_indices:
            self._assign_distance_based(pairs)
            return
        default_channel_index = channel_indices[0]

        sfs = sorted({sf for access in self.node_sf_access.values() for sf in access})
        if not sfs:
            sfs = [7, 8, 9, 10, 11, 12]
        sf_order_map = {sf: position for position, sf in enumerate(sfs)}
        airtimes = self.sf_airtimes or self._compute_sf_airtimes(simulator, sfs)
        sf_channel_caps = self.cluster_sf_channel_capacity

        duty_manager = getattr(simulator, "duty_cycle_manager", None)
        duty_cycle = float(getattr(duty_manager, "duty_cycle", 0.0)) if duty_manager is not None else 0.0
        duty_limit = duty_cycle if duty_cycle > 0.0 else float("inf")
        channel_remaining = {idx: duty_limit for idx in channel_indices}

        cluster_nodes: dict[int, list[tuple[_NodeDistance, list[int]]]] = {}
        fallback_pairs: list[_NodeDistance] = []
        for entry in pairs:
            node = entry.node
            node_id = getattr(node, "id", id(node))
            cluster_id = self.node_clusters.get(node_id)
            accessible = self.node_sf_access.get(node_id) or []
            if cluster_id is None or not accessible:
                fallback_pairs.append(entry)
                continue
            cluster_nodes.setdefault(cluster_id, []).append((entry, accessible))

        if not cluster_nodes:
            self._assign_distance_based(pairs)
            return

        ordered_clusters = sorted(self.clusters, key=lambda c: c.pdr_target, reverse=True)
        channel_pos = 0

        for cluster in ordered_clusters:
            entries = cluster_nodes.get(cluster.cluster_id)
            if not entries:
                continue
            cap_limit = self.cluster_capacity_limits.get(cluster.cluster_id)
            if cap_limit is None or cap_limit <= 0.0:
                cluster_remaining = float("inf")
            else:
                cluster_remaining = float(cap_limit)
            for entry, accessible in entries:
                if not accessible:
                    fallback_pairs.append(entry)
                    continue
                assigned = False
                for sf in accessible:
                    airtime = airtimes.get(sf, 0.0)
                    load = cluster.arrival_rate * airtime
                    if load < 0.0:
                        load = 0.0
                    if cluster_remaining < load - 1e-12:
                        continue
                    for offset in range(len(channel_indices)):
                        idx = channel_indices[(channel_pos + offset) % len(channel_indices)]
                        remaining = channel_remaining.get(idx, float("inf"))
                        if load <= remaining + 1e-12:
                            channel_pos = (channel_pos + offset + 1) % len(channel_indices)
                            node = entry.node
                            node.sf = sf
                            channel_obj = channel_map.get(idx, channel_map.get(default_channel_index))
                            if channel_obj is not None:
                                node.channel = channel_obj
                            sf_index = sf_order_map.get(sf, len(sfs) - 1)
                            node.tx_power = self._assign_tx_power(sf_index)
                            if load > 0.0:
                                channel_remaining[idx] = max(0.0, remaining - load)
                                if cluster_remaining < float("inf"):
                                    cluster_remaining = max(0.0, cluster_remaining - load)
                            assigned = True
                            break
                    if assigned:
                        break
                if not assigned:
                    fallback_pairs.append(entry)

        self._assign_distance_based(fallback_pairs)

    def _apply_aimi_like(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        if not self.clusters or not self.node_sf_access:
            self._assign_distance_based(pairs)
            return

        channel_map: dict[int, object | None] = {}
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            for index, channel in enumerate(getattr(multichannel, "channels", []) or []):
                channel_index = getattr(channel, "channel_index", index)
                channel_map[channel_index] = channel
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channel_index = getattr(base_channel, "channel_index", 0)
            channel_map.setdefault(channel_index, base_channel)
        if not channel_map:
            channel_map[0] = base_channel
        channel_indices = sorted(channel_map)
        if not channel_indices:
            self._assign_distance_based(pairs)
            return
        default_channel_index = channel_indices[0]

        sfs = sorted({sf for access in self.node_sf_access.values() for sf in access})
        if not sfs:
            sfs = [7, 8, 9, 10, 11, 12]
        sf_order_map = {sf: position for position, sf in enumerate(sfs)}
        airtimes = self.sf_airtimes or self._compute_sf_airtimes(simulator, sfs)
        sf_channel_caps = self.cluster_sf_channel_capacity

        duty_manager = getattr(simulator, "duty_cycle_manager", None)
        duty_cycle = float(getattr(duty_manager, "duty_cycle", 0.0)) if duty_manager is not None else 0.0
        duty_limit = duty_cycle if duty_cycle > 0.0 else float("inf")

        cluster_nodes: dict[int, list[tuple[_NodeDistance, list[int]]]] = {}
        fallback_pairs: list[_NodeDistance] = []
        for entry in pairs:
            node = entry.node
            node_id = getattr(node, "id", id(node))
            cluster_id = self.node_clusters.get(node_id)
            accessible = self.node_sf_access.get(node_id) or []
            if cluster_id is None or not accessible:
                fallback_pairs.append(entry)
                continue
            cluster_nodes.setdefault(cluster_id, []).append((entry, accessible))

        if not cluster_nodes:
            self._assign_distance_based(pairs)
            return

        ordered_clusters = sorted(self.clusters, key=lambda c: c.pdr_target, reverse=True)
        channel_count = len(channel_indices)
        allocations: dict[int, int] = {cluster.cluster_id: 0 for cluster in ordered_clusters}
        remaining = channel_count
        for cluster in ordered_clusters:
            if remaining <= 0:
                break
            allocations[cluster.cluster_id] += 1
            remaining -= 1
        if remaining > 0 and ordered_clusters:
            total_share = sum(cluster.device_share for cluster in ordered_clusters)
            fractional: list[tuple[float, int]] = []
            if total_share > 0.0:
                for cluster in ordered_clusters:
                    share = cluster.device_share / total_share
                    extra = share * remaining
                    base = math.floor(extra)
                    allocations[cluster.cluster_id] += base
                    fractional.append((extra - base, cluster.cluster_id))
            else:
                even = remaining / len(ordered_clusters)
                for cluster in ordered_clusters:
                    extra = even
                    base = math.floor(extra)
                    allocations[cluster.cluster_id] += base
                    fractional.append((extra - base, cluster.cluster_id))
            allocated = sum(allocations.values())
            leftover = channel_count - allocated
            if leftover > 0:
                fractional.sort(reverse=True)
                for _, cluster_id in fractional:
                    if leftover <= 0:
                        break
                    allocations[cluster_id] += 1
                    leftover -= 1

        channel_partitions: dict[int, list[int]] = {}
        position = 0
        for cluster in ordered_clusters:
            count = allocations.get(cluster.cluster_id, 0)
            if count < 0:
                count = 0
            assigned = channel_indices[position : position + count]
            channel_partitions[cluster.cluster_id] = list(assigned)
            position += len(assigned)
        if position < channel_count and ordered_clusters:
            extra_indices = channel_indices[position:]
            if extra_indices:
                first_id = ordered_clusters[0].cluster_id
                channel_partitions.setdefault(first_id, []).extend(extra_indices)

        for cluster in ordered_clusters:
            entries = cluster_nodes.get(cluster.cluster_id)
            if not entries:
                continue
            reserved = channel_partitions.get(cluster.cluster_id) or [channel_indices[-1]]
            per_channel_remaining = {idx: duty_limit for idx in reserved}
            cap_limit = self.cluster_capacity_limits.get(cluster.cluster_id)
            if cap_limit is None or cap_limit <= 0.0:
                cluster_remaining = float("inf")
            else:
                cluster_remaining = float(cap_limit)
            channel_pos = 0
            for entry, accessible in entries:
                if not accessible:
                    fallback_pairs.append(entry)
                    continue
                assigned = False
                for sf in accessible:
                    airtime = airtimes.get(sf, 0.0)
                    load = cluster.arrival_rate * airtime
                    if load < 0.0:
                        load = 0.0
                    if cluster_remaining < load - 1e-12:
                        continue
                    for offset in range(len(reserved)):
                        idx = reserved[(channel_pos + offset) % len(reserved)]
                        remaining = per_channel_remaining.get(idx, float("inf"))
                        if load <= remaining + 1e-12:
                            channel_pos = (channel_pos + offset + 1) % len(reserved)
                            node = entry.node
                            node.sf = sf
                            channel_obj = channel_map.get(idx, channel_map.get(default_channel_index))
                            if channel_obj is not None:
                                node.channel = channel_obj
                            sf_index = sf_order_map.get(sf, len(sfs) - 1)
                            node.tx_power = self._assign_tx_power(sf_index)
                            if load > 0.0:
                                per_channel_remaining[idx] = max(0.0, remaining - load)
                                if cluster_remaining < float("inf"):
                                    cluster_remaining = max(0.0, cluster_remaining - load)
                            assigned = True
                            break
                    if assigned:
                        break
                if not assigned:
                    fallback_pairs.append(entry)

        self._assign_distance_based(fallback_pairs)

    # --- Implémentations MixRA ---------------------------------------------
    def _assign_distance_based(self, entries: Sequence[_NodeDistance]) -> None:
        if not entries:
            return
        sfs = [7, 8, 9, 10, 11, 12]
        chunk_size = max(1, math.ceil(len(entries) / len(sfs)))
        for index, entry in enumerate(entries):
            sf_index = min(index // chunk_size, len(sfs) - 1)
            sf = sfs[sf_index]
            blocked_min_sf = getattr(entry.node, "_qos_blocked_min_sf", None)
            if blocked_min_sf is not None:
                for candidate in sfs[sf_index:]:
                    if candidate > blocked_min_sf:
                        sf = candidate
                        break
                else:
                    sf = sfs[-1]
                sf_index = sfs.index(sf)
            entry.node.sf = sf
            entry.node.tx_power = self._assign_tx_power(sf_index)
            if getattr(entry.node, "_qos_blocked_channel", False):
                entry.node.channel = None
                self._flag_out_of_service(entry.node, "blocked_channel")

    def _flag_out_of_service(self, node, reason: str) -> None:
        if getattr(node, "out_of_service", False):
            return
        setattr(node, "out_of_service", True)
        self.out_of_service_queue.append((node.id, reason))

    @staticmethod
    def _solve_mixra_greedy(
        bounds: Sequence[tuple[float, float]],
        cluster_var_indices: dict[int, list[int]],
        sf_limits: dict[tuple[int, int], float],
        var_data: Sequence[dict[str, float | int]],
        throughput_coeffs: Sequence[float],
        cluster_caps: dict[int, float],
        sf_channel_caps: dict[int, dict[int, dict[int, float]]],
        duty_cycle: float | None,
    ) -> list[float] | None:
        solution = [0.0] * len(var_data)
        sf_remaining = {key: limit for key, limit in sf_limits.items()}
        channel_remaining: dict[int, float] = {}
        if duty_cycle is not None and duty_cycle > 0.0:
            for data in var_data:
                channel_index = int(data["channel"])
                channel_remaining.setdefault(channel_index, float(duty_cycle))
        cluster_remaining = {
            cluster_id: float(capacity) for cluster_id, capacity in cluster_caps.items() if capacity > 0.0
        }
        combo_remaining: dict[tuple[int, int, int], float] = {
            (cluster_id, sf, channel_index): float(cap)
            for cluster_id, sf_map in sf_channel_caps.items()
            for sf, channel_map in sf_map.items()
            for channel_index, cap in channel_map.items()
        }

        for cluster_id, indices in cluster_var_indices.items():
            remaining_share = 1.0
            ordered = sorted(indices, key=lambda idx: throughput_coeffs[idx], reverse=True)
            for idx in ordered:
                if remaining_share <= 0.0:
                    break
                upper = bounds[idx][1]
                if upper <= 0.0:
                    continue
                data = var_data[idx]
                sf = int(data["sf"])
                channel_index = int(data["channel"])
                load_coeff = float(data["load_coeff"])
                share_limit = min(upper, remaining_share)
                sf_key = (cluster_id, sf)
                sf_cap = sf_remaining.get(sf_key)
                if sf_cap is not None:
                    share_limit = min(share_limit, sf_cap)
                if duty_cycle is not None and duty_cycle > 0.0:
                    channel_cap = channel_remaining.get(channel_index)
                    if channel_cap is not None and load_coeff > 0.0:
                        share_limit = min(share_limit, channel_cap / load_coeff)
                combo_cap = combo_remaining.get((cluster_id, sf, channel_index))
                if combo_cap is not None:
                    if combo_cap <= 0.0:
                        continue
                    if load_coeff > 0.0:
                        share_limit = min(share_limit, combo_cap / load_coeff)
                cluster_cap = cluster_remaining.get(cluster_id)
                if cluster_cap is not None and load_coeff > 0.0:
                    share_limit = min(share_limit, cluster_cap / load_coeff)
                if share_limit <= 0.0:
                    continue
                solution[idx] = share_limit
                remaining_share -= share_limit
                if sf_cap is not None:
                    sf_remaining[sf_key] = max(0.0, sf_cap - share_limit)
                if duty_cycle is not None and duty_cycle > 0.0:
                    channel_cap = channel_remaining.get(channel_index)
                    if channel_cap is not None and load_coeff > 0.0:
                        channel_remaining[channel_index] = max(
                            0.0, channel_cap - share_limit * load_coeff
                        )
                if combo_cap is not None and load_coeff > 0.0:
                    combo_remaining[(cluster_id, sf, channel_index)] = max(
                        0.0, combo_cap - share_limit * load_coeff
                    )
                if cluster_cap is not None and load_coeff > 0.0:
                    cluster_remaining[cluster_id] = max(0.0, cluster_cap - share_limit * load_coeff)
            if remaining_share > 1e-6:
                return None
        return solution

    def _apply_mixra_opt(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        if not self.clusters or not self.node_sf_access:
            self._assign_distance_based(pairs)
            return

        # Recensement des canaux disponibles.
        channel_map: dict[int, object | None] = {}
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            for index, channel in enumerate(getattr(multichannel, "channels", []) or []):
                channel_index = getattr(channel, "channel_index", index)
                channel_map[channel_index] = channel
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channel_index = getattr(base_channel, "channel_index", 0)
            channel_map.setdefault(channel_index, base_channel)
        if not channel_map:
            channel_map[0] = base_channel
        channel_indices = sorted(channel_map)

        channel_cluster_limit = self.max_clusters_per_channel
        sf_cluster_limit = self.max_clusters_per_min_sf
        channel_cluster_usage: dict[int, set[int]] = {idx: set() for idx in channel_indices}
        sf_cluster_usage: dict[int, set[int]] = defaultdict(set)

        cluster_min_sf_map: dict[int, int] = {}
        for cluster_id, sf_counts in self.cluster_d_matrix.items():
            for sf in sorted(sf_counts):
                if sf_counts[sf] > 0:
                    cluster_min_sf_map[cluster_id] = sf
                    break

        # Classement des nœuds par cluster en conservant l'ordre par distance.
        cluster_nodes: dict[int, list[tuple[_NodeDistance, list[int]]]] = {}
        fallback_pairs: list[_NodeDistance] = []
        for entry in pairs:
            node = entry.node
            node_id = getattr(node, "id", id(node))
            cluster_id = self.node_clusters.get(node_id)
            accessible = self.node_sf_access.get(node_id) or []
            if cluster_id is None or not accessible:
                fallback_pairs.append(entry)
                continue
            cluster_nodes.setdefault(cluster_id, []).append((entry, accessible))

        if not cluster_nodes:
            self._assign_distance_based(pairs)
            return

        # Préparation des paramètres d'optimisation.
        sfs = sorted({sf for access in self.node_sf_access.values() for sf in access}) or [7, 8, 9, 10, 11, 12]
        airtimes = self.sf_airtimes or self._compute_sf_airtimes(simulator, sfs)
        payload_bits = float(getattr(simulator, "payload_size_bytes", 20) * 8)
        cluster_lookup = {cluster.cluster_id: cluster for cluster in self.clusters}
        cluster_caps = {
            cluster_id: cap
            for cluster_id, cap in self.cluster_capacity_limits.items()
            if cap is not None and cap > 0.0
        }
        sf_channel_caps = self.cluster_sf_channel_capacity

        var_data: list[dict[str, float | int]] = []
        throughput_coeffs: list[float] = []
        bounds: list[tuple[float, float]] = []
        cluster_var_indices: dict[int, list[int]] = {}
        sf_cluster_indices: dict[tuple[int, int], list[int]] = {}
        channel_var_indices: dict[int, list[int]] = {}
        sf_limits: dict[tuple[int, int], float] = {}
        cluster_sizes: dict[int, int] = {}

        for cluster_id, entries in cluster_nodes.items():
            cluster = cluster_lookup.get(cluster_id)
            if cluster is None:
                fallback_pairs.extend(entry for entry, _ in entries)
                continue
            node_count = len(entries)
            if node_count == 0:
                fallback_pairs.extend(entry for entry, _ in entries)
                continue

            min_sf = cluster_min_sf_map.get(cluster_id)
            if sf_cluster_limit is not None and min_sf is not None:
                sf_set = sf_cluster_usage[min_sf]
                if cluster_id not in sf_set and len(sf_set) >= sf_cluster_limit:
                    for entry, _ in entries:
                        setattr(entry.node, "_qos_blocked_min_sf", min_sf)
                        setattr(entry.node, "_qos_blocked_channel", True)
                        entry.node.channel = None
                        self._flag_out_of_service(entry.node, "sf_cluster_limit")
                    fallback_pairs.extend(entry for entry, _ in entries)
                    continue
            cluster_sizes[cluster_id] = node_count
            lambda_k = float(cluster.arrival_rate)
            pdr_k = float(cluster.pdr_target)
            accessible_counts: dict[int, int] = {sf: 0 for sf in sfs}
            for _, access in entries:
                for sf in access:
                    if sf in accessible_counts:
                        accessible_counts[sf] += 1

            for sf in sfs:
                tau = float(airtimes.get(sf, 0.0))
                access_count = accessible_counts.get(sf, 0)
                if tau <= 0.0 or access_count <= 0:
                    continue
                load_coeff = node_count * lambda_k * tau
                if load_coeff <= 0.0:
                    continue
                max_share = access_count / node_count
                sf_limits[(cluster_id, sf)] = max_share
                throughput = node_count * lambda_k * payload_bits * pdr_k
                if tau > 0.0:
                    throughput /= tau
                cap_limit = cluster_caps.get(cluster_id)

                for channel_index in channel_indices:
                    upper = min(1.0, max_share)
                    if cap_limit is not None:
                        upper = min(upper, cap_limit / load_coeff)
                    combo_cap = (
                        sf_channel_caps.get(cluster_id, {})
                        .get(sf, {})
                        .get(channel_index)
                    )
                    if combo_cap is not None:
                        if combo_cap <= 0.0:
                            continue
                        if load_coeff > 0.0:
                            upper = min(upper, combo_cap / load_coeff)
                    if upper <= 0.0:
                        continue
                    idx = len(var_data)
                    var_data.append(
                        {
                            "cluster_id": cluster_id,
                            "sf": sf,
                            "channel": channel_index,
                            "load_coeff": load_coeff,
                        }
                    )
                    throughput_coeffs.append(throughput)
                    bounds.append((0.0, upper))
                    cluster_var_indices.setdefault(cluster_id, []).append(idx)
                    sf_cluster_indices.setdefault((cluster_id, sf), []).append(idx)
                    channel_var_indices.setdefault(channel_index, []).append(idx)

        # Abandon si aucun couple optimisable n'est disponible.
        if not var_data:
            self._assign_distance_based(pairs)
            return

        # Chaque cluster doit pouvoir satisfaire la contrainte de somme.
        for cluster_id, indices in cluster_var_indices.items():
            total_capacity = sum(bounds[idx][1] for idx in indices)
            if total_capacity < 1.0 - 1e-9:
                self._assign_distance_based(pairs)
                return

        # Construction d'un point initial respectant les bornes et la contrainte d'égalité.
        x0 = [0.0] * len(var_data)
        for cluster_id, indices in cluster_var_indices.items():
            remaining = 1.0
            for idx in sorted(indices, key=lambda i: bounds[i][1], reverse=True):
                upper = bounds[idx][1]
                value = min(upper, remaining)
                x0[idx] = value
                remaining -= value
                if remaining <= 1e-9:
                    break
            if remaining > 1e-6:
                self._assign_distance_based(pairs)
                return

        def objective(values):
            total = 0.0
            for coeff, value in zip(throughput_coeffs, values):
                if value <= 0.0:
                    continue
                total += coeff * float(value)
            return -total

        constraints = []
        for cluster_id, indices in cluster_var_indices.items():
            idxs = tuple(indices)
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda values, idxs=idxs: sum(float(values[i]) for i in idxs) - 1.0,
                }
            )
            cap_limit = cluster_caps.get(cluster_id)
            if cap_limit is not None:
                terms = tuple((i, float(var_data[i]["load_coeff"])) for i in idxs)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda values, terms=terms, limit=cap_limit: limit
                        - sum(float(values[i]) * coeff for i, coeff in terms),
                    }
                )

        for (cluster_id, sf), indices in sf_cluster_indices.items():
            limit = sf_limits.get((cluster_id, sf))
            if limit is None:
                continue
            idxs = tuple(indices)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda values, idxs=idxs, limit=limit: limit
                    - sum(float(values[i]) for i in idxs),
                }
            )

        duty_manager = getattr(simulator, "duty_cycle_manager", None)
        duty_cycle = getattr(duty_manager, "duty_cycle", None) if duty_manager is not None else None
        if duty_cycle is not None and duty_cycle > 0.0:
            for channel_id, indices in channel_var_indices.items():
                terms = tuple((i, float(var_data[i]["load_coeff"])) for i in indices)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda values, terms=terms, limit=duty_cycle: limit
                        - sum(float(values[i]) * coeff for i, coeff in terms),
                    }
                )

        if minimize is None:
            solution = self._solve_mixra_greedy(
                bounds,
                cluster_var_indices,
                sf_limits,
                var_data,
                throughput_coeffs,
                cluster_caps,
                self.cluster_sf_channel_capacity,
                duty_cycle,
            )
            if solution is None:
                self._assign_distance_based(pairs)
                return
        else:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )

            if not result.success or not result.x.size:
                self._assign_distance_based(pairs)
                return

            solution = [max(0.0, min(bounds[idx][1], float(value))) for idx, value in enumerate(result.x)]

        # Normalisation par cluster pour éviter les dérives numériques.
        for cluster_id, indices in cluster_var_indices.items():
            total = sum(solution[idx] for idx in indices)
            if total <= 0.0:
                continue
            for idx in indices:
                solution[idx] /= total

        sf_order = [7, 8, 9, 10, 11, 12]
        sf_order_map = {sf: position for position, sf in enumerate(sf_order)}
        default_channel_index = channel_indices[0]

        for cluster_id, entries in cluster_nodes.items():
            indices = cluster_var_indices.get(cluster_id)
            if not indices:
                fallback_pairs.extend(entry for entry, _ in entries)
                continue
            node_count = cluster_sizes.get(cluster_id, len(entries))
            counts: dict[int, int] = {}
            residuals: list[tuple[float, int]] = []
            max_nodes_per_index: dict[int, int] = {}
            min_sf = cluster_min_sf_map.get(cluster_id)
            for idx in indices:
                share = solution[idx]
                target = share * node_count
                base = math.floor(target)
                max_nodes = int(math.floor(bounds[idx][1] * node_count + 1e-9))
                if max_nodes < 0:
                    max_nodes = 0
                load_coeff = float(var_data[idx]["load_coeff"])
                if node_count > 0 and load_coeff > 0.0:
                    per_node_load = load_coeff / node_count
                    if duty_cycle is not None and duty_cycle > 0.0:
                        channel_limit = int(math.floor((duty_cycle + 1e-9) / per_node_load))
                        if channel_limit < max_nodes:
                            max_nodes = channel_limit
                    cluster_cap = cluster_caps.get(cluster_id)
                    if cluster_cap is not None:
                        cluster_limit = int(math.floor((cluster_cap + 1e-9) / per_node_load))
                        if cluster_limit < max_nodes:
                            max_nodes = cluster_limit
                    combo_cap = (
                        sf_channel_caps.get(cluster_id, {})
                        .get(sf, {})
                        .get(channel_index)
                    )
                    if combo_cap is not None:
                        combo_limit = int(math.floor((combo_cap + 1e-9) / per_node_load))
                        if combo_limit < max_nodes:
                            max_nodes = combo_limit
                if base > max_nodes:
                    base = max_nodes
                counts[idx] = int(base)
                residual = max(target - base, 0.0)
                residuals.append((residual, idx))
                max_nodes_per_index[idx] = max_nodes
            remaining = node_count - sum(counts.values())
            if remaining > 0:
                residuals.sort(reverse=True)
                for _, idx in residuals:
                    if remaining <= 0:
                        break
                    max_nodes = max_nodes_per_index.get(idx, node_count)
                    if counts[idx] >= max_nodes:
                        continue
                    counts[idx] += 1
                    remaining -= 1

            available = list(entries)
            assignments: dict[int, list[tuple[_NodeDistance, list[int]]]] = {idx: [] for idx in indices}
            combo_remaining_load: dict[tuple[int, int], float] = {}
            for idx in indices:
                sf_val = int(var_data[idx]["sf"])
                channel_val = int(var_data[idx]["channel"])
                cap_val = (
                    sf_channel_caps.get(cluster_id, {})
                    .get(sf_val, {})
                    .get(channel_val)
                )
                if cap_val is None:
                    combo_remaining_load[(sf_val, channel_val)] = float("inf")
                else:
                    combo_remaining_load[(sf_val, channel_val)] = float(max(cap_val, 0.0))

            for idx in sorted(indices, key=lambda i: (solution[i], -var_data[i]["sf"]), reverse=True):
                target = counts.get(idx, 0)
                if target <= 0:
                    continue
                sf = int(var_data[idx]["sf"])
                for entry, access in list(available):
                    if sf not in access:
                        continue
                    assignments[idx].append((entry, access))
                    available.remove((entry, access))
                    if len(assignments[idx]) >= target:
                        break

            for idx in indices:
                target = counts.get(idx, 0)
                allocated = len(assignments.get(idx, []))
                if allocated >= target:
                    continue
                sf = int(var_data[idx]["sf"])
                for entry, access in list(available):
                    if sf not in access:
                        continue
                    assignments[idx].append((entry, access))
                    available.remove((entry, access))
                    allocated += 1
                    if allocated >= target:
                        break

            for idx in indices:
                assigned = assignments.get(idx, [])
                if not assigned:
                    continue
                sf = int(var_data[idx]["sf"])
                channel_index = int(var_data[idx]["channel"])
                if channel_cluster_limit is not None:
                    channel_set = channel_cluster_usage.setdefault(channel_index, set())
                    if cluster_id not in channel_set and len(channel_set) >= channel_cluster_limit:
                        for entry, _ in assigned:
                            setattr(entry.node, "_qos_blocked_channel", True)
                            entry.node.channel = None
                            self._flag_out_of_service(entry.node, "channel_cluster_limit")
                            fallback_pairs.append(entry)
                        continue
                if sf_cluster_limit is not None and min_sf is not None and cluster_id not in sf_cluster_usage[min_sf]:
                    sf_cluster_usage[min_sf].add(cluster_id)
                channel_obj = channel_map.get(channel_index, channel_map.get(default_channel_index))
                sf_index = sf_order_map.get(sf, len(sf_order) - 1)
                load_coeff = float(var_data[idx]["load_coeff"])
                per_node_load = load_coeff / node_count if node_count > 0 else 0.0
                channel_cluster_usage.setdefault(channel_index, set()).add(cluster_id)
                for entry, _ in assigned:
                    remaining_cap = combo_remaining_load.get((sf, channel_index), float("inf"))
                    if per_node_load > 0.0 and remaining_cap < per_node_load - 1e-9:
                        break
                    node = entry.node
                    node.sf = sf
                    if channel_obj is not None:
                        node.channel = channel_obj
                    node.tx_power = self._assign_tx_power(sf_index)
                    if per_node_load > 0.0:
                        combo_remaining_load[(sf, channel_index)] = max(
                            0.0, remaining_cap - per_node_load
                        )

            for entry, access in available:
                if not access:
                    fallback_pairs.append(entry)
                    continue
                sf = access[0]
                channel_obj = channel_map.get(default_channel_index)
                sf_index = sf_order_map.get(sf, len(sf_order) - 1)
                node = entry.node
                node.sf = sf
                if channel_obj is not None:
                    node.channel = channel_obj
                node.tx_power = self._assign_tx_power(sf_index)

        self._assign_distance_based(fallback_pairs)

    def _apply_mixra_h(self, simulator) -> None:
        pairs = self._sorted_distances(simulator)
        if not pairs:
            return
        if not self.clusters or not self.node_sf_access:
            self._assign_distance_based(pairs)
            return

        channel_map: dict[int, object | None] = {}
        multichannel = getattr(simulator, "multichannel", None)
        if multichannel is not None:
            for index, channel in enumerate(getattr(multichannel, "channels", []) or []):
                channel_index = getattr(channel, "channel_index", index)
                channel_map[channel_index] = channel
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channel_index = getattr(base_channel, "channel_index", 0)
            channel_map.setdefault(channel_index, base_channel)
        if not channel_map:
            channel_map[0] = base_channel
        channel_indices = sorted(channel_map)
        default_channel_index = channel_indices[0]

        channel_cluster_limit = self.max_clusters_per_channel
        sf_cluster_limit = self.max_clusters_per_min_sf
        channel_cluster_usage: dict[int, set[int]] = {idx: set() for idx in channel_indices}
        sf_cluster_usage: dict[int, set[int]] = defaultdict(set)

        sfs = sorted({sf for access in self.node_sf_access.values() for sf in access})
        if not sfs:
            sfs = [7, 8, 9, 10, 11, 12]
        sf_order_map = {sf: position for position, sf in enumerate(sfs)}

        airtimes = self.sf_airtimes or self._compute_sf_airtimes(simulator, sfs)
        sf_channel_caps = self.cluster_sf_channel_capacity

        cluster_min_sf_map: dict[int, int] = {}
        for cluster_id, sf_counts in self.cluster_d_matrix.items():
            for sf in sorted(sf_counts):
                if sf_counts[sf] > 0:
                    cluster_min_sf_map[cluster_id] = sf
                    break

        cluster_nodes: dict[int, dict[int, deque[tuple[_NodeDistance, list[int]]]]] = {}
        fallback_pairs: list[_NodeDistance] = []
        for entry in pairs:
            node = entry.node
            node_id = getattr(node, "id", id(node))
            cluster_id = self.node_clusters.get(node_id)
            accessible = self.node_sf_access.get(node_id) or []
            if cluster_id is None or not accessible:
                fallback_pairs.append(entry)
                continue
            min_sf = accessible[0]
            cluster_nodes.setdefault(cluster_id, {sf: deque() for sf in sfs})
            if min_sf not in cluster_nodes[cluster_id]:
                cluster_nodes[cluster_id][min_sf] = deque()
            cluster_nodes[cluster_id][min_sf].append((entry, accessible))

        if not cluster_nodes:
            self._assign_distance_based(pairs)
            return

        ordered_clusters = sorted(self.clusters, key=lambda c: c.pdr_target, reverse=True)

        for cluster in ordered_clusters:
            cluster_id = cluster.cluster_id
            buckets = cluster_nodes.get(cluster_id)
            if not buckets:
                continue
            min_sf = cluster_min_sf_map.get(cluster_id)
            if sf_cluster_limit is not None and min_sf is not None:
                sf_set = sf_cluster_usage[min_sf]
                if cluster_id not in sf_set and len(sf_set) >= sf_cluster_limit:
                    for queue in buckets.values():
                        while queue:
                            entry, _ = queue.popleft()
                            setattr(entry.node, "_qos_blocked_min_sf", min_sf)
                            setattr(entry.node, "_qos_blocked_channel", True)
                            entry.node.channel = None
                            self._flag_out_of_service(entry.node, "sf_cluster_limit")
                            fallback_pairs.append(entry)
                    continue
            capacity_limit = self.cluster_capacity_limits.get(cluster_id)
            if capacity_limit is None or capacity_limit <= 0.0:
                per_channel_capacity = float("inf")
                cluster_remaining = float("inf")
            else:
                per_channel_capacity = float(capacity_limit)
                cluster_remaining = float(capacity_limit)
            channel_remaining = {idx: per_channel_capacity for idx in channel_indices}
            channel_pos = 0
            cluster_exhausted = False
            combo_remaining = {
                (sf_key, channel_idx): float(max(cap, 0.0))
                for sf_key, channel_map in sf_channel_caps.get(cluster_id, {}).items()
                for channel_idx, cap in channel_map.items()
            }

            for sf_index, sf in enumerate(sfs):
                queue = buckets.get(sf)
                if not queue:
                    continue
                load_per_node = cluster.arrival_rate * airtimes.get(sf, 0.0)
                while queue:
                    if cluster_remaining <= 1e-9:
                        cluster_remaining = 0.0
                        cluster_exhausted = True
                        break
                    if load_per_node <= 0.0:
                        entry, access = queue.popleft()
                        target_sf = sf if sf in access else (access[0] if access else sf)
                        sf_pos = sf_order_map.get(target_sf, len(sfs) - 1)
                        channel_idx = channel_indices[min(channel_pos, len(channel_indices) - 1)]
                        channel_obj = channel_map.get(channel_idx, channel_map.get(default_channel_index))
                        node = entry.node
                        node.sf = target_sf
                        if channel_obj is not None:
                            node.channel = channel_obj
                        node.tx_power = self._assign_tx_power(sf_pos)
                        continue

                    if channel_pos >= len(channel_indices):
                        if sf_index + 1 < len(sfs):
                            next_sf = sfs[sf_index + 1]
                            next_queue = buckets.setdefault(next_sf, deque())
                            while queue:
                                entry, access = queue.popleft()
                                if next_sf in access:
                                    next_queue.append((entry, access))
                                else:
                                    setattr(entry.node, "_qos_blocked_channel", True)
                                    entry.node.channel = None
                                    self._flag_out_of_service(entry.node, "channel_exhausted")
                                    fallback_pairs.append(entry)
                            break
                        else:
                            while queue:
                                entry, access = queue.popleft()
                                if not access:
                                    fallback_pairs.append(entry)
                                    continue
                                chosen_sf = sf if sf in access else access[-1]
                                sf_pos = sf_order_map.get(chosen_sf, len(sfs) - 1)
                                node = entry.node
                                node.sf = chosen_sf
                                node.tx_power = self._assign_tx_power(sf_pos)
                                setattr(node, "_qos_blocked_channel", True)
                                node.channel = None
                                self._flag_out_of_service(node, "channel_exhausted")
                                fallback_pairs.append(entry)
                            break

                    if channel_pos >= len(channel_indices):
                        break

                    channel_idx = channel_indices[channel_pos]
                    if channel_cluster_limit is not None:
                        channel_set = channel_cluster_usage.setdefault(channel_idx, set())
                        if cluster_id not in channel_set and len(channel_set) >= channel_cluster_limit:
                            channel_pos += 1
                            continue
                    remaining = channel_remaining.get(channel_idx, float("inf"))
                    combo_key = (sf, channel_idx)
                    combo_cap = combo_remaining.get(combo_key, float("inf"))
                    effective_remaining = min(remaining, cluster_remaining, combo_cap)
                    if effective_remaining <= load_per_node - 1e-9:
                        if combo_key in combo_remaining:
                            combo_remaining[combo_key] = max(0.0, combo_cap)
                        channel_remaining[channel_idx] = max(0.0, remaining)
                        channel_pos += 1
                        continue

                    entry, access = queue.popleft()
                    if sf not in access:
                        # Recherche du prochain SF accessible pour cet équipement.
                        promoted_sf = None
                        for candidate in access:
                            if candidate >= sf:
                                promoted_sf = candidate
                                break
                        if promoted_sf is None:
                            fallback_pairs.append(entry)
                        elif promoted_sf == sf:
                            queue.appendleft((entry, access))
                        else:
                            buckets.setdefault(promoted_sf, deque()).append((entry, access))
                        continue

                    channel_obj = channel_map.get(channel_idx, channel_map.get(default_channel_index))
                    node = entry.node
                    node.sf = sf
                    if channel_obj is not None:
                        node.channel = channel_obj
                    node.tx_power = self._assign_tx_power(sf_order_map.get(sf, len(sfs) - 1))
                    if min_sf is not None and sf_cluster_limit is not None:
                        sf_cluster_usage[min_sf].add(cluster_id)
                    channel_cluster_usage.setdefault(channel_idx, set()).add(cluster_id)
                    channel_remaining[channel_idx] = max(0.0, remaining - load_per_node)
                    if combo_key in combo_remaining:
                        combo_remaining[combo_key] = max(0.0, combo_cap - load_per_node)
                    cluster_remaining = max(0.0, cluster_remaining - load_per_node)
                    if channel_remaining[channel_idx] <= 1e-9 or (
                        combo_key in combo_remaining and combo_remaining[combo_key] <= 1e-9
                    ):
                        channel_remaining[channel_idx] = max(0.0, channel_remaining[channel_idx])
                        if combo_key in combo_remaining:
                            combo_remaining[combo_key] = max(0.0, combo_remaining[combo_key])
                        channel_pos += 1
                    if cluster_remaining <= 1e-9:
                        cluster_remaining = 0.0
                        cluster_exhausted = True
                        break

                if cluster_exhausted:
                    break

            for sf in sfs:
                queue = buckets.get(sf)
                if not queue:
                    continue
                while queue:
                    entry, _ = queue.popleft()
                    setattr(entry.node, "_qos_blocked_channel", True)
                    entry.node.channel = None
                    self._flag_out_of_service(entry.node, "capacity_exhausted")
                    fallback_pairs.append(entry)

        self._assign_distance_based(fallback_pairs)

