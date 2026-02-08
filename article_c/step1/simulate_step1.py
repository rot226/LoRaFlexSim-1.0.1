"""Simulation de l'étape 1 avec proxys ADR/MixRA.

Les implémentations ci-dessous fournissent des heuristiques simples et
reproductibles pour ADR, MixRA-H et MixRA-Opt. Elles servent de substituts
cohérents lorsque les formules exactes ne sont pas disponibles.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
from statistics import mean
from time import perf_counter
from typing import Iterable

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.interference import Signal, compute_co_sf_overlaps
from article_c.common.lora_phy import coding_rate_to_cr, compute_airtime
from article_c.common.metrics import (
    energy_per_success_bit,
    mean_toa_s,
    packet_delivery_ratio,
)
from article_c.common.propagation import sample_fading_db
from article_c.common.utils import assign_clusters, generate_traffic_times

SF_VALUES = (7, 8, 9, 10, 11, 12)
SF_INDEX = {sf: idx for idx, sf in enumerate(SF_VALUES)}
MIXRA_OPT_BUDGET_BY_SIZE = {
    80: 50000,
    160: 100000,
    320: 200000,
    640: 400000,
    1280: 800000,
}
MIXRA_OPT_BUDGET_PER_NODE = 50000 / 80
MIXRA_OPT_EMERGENCY_TIMEOUT_S = 300.0

# Seuils proxy pour SNR/RSSI (inspirés d'ordres de grandeur LoRaWAN).
SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
RSSI_THRESHOLDS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}
ADR_MARGIN_BUFFER = 0.6
MIXRA_CLUSTER_STRENGTH = 0.35

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeLink:
    snr: float
    rssi: float
    qos_margin: float
    snr_margins: tuple[float, ...]
    rssi_margins: tuple[float, ...]
    distance_km: float
    tx_power_dbm: float
    quality_db: float


@dataclass
class Step1Result:
    sent: int
    received: int
    energy_per_success_bit: float
    mean_toa_s: float
    node_clusters: list[str]
    node_received: list[bool]
    toa_s_by_node: list[float]
    packet_ids: list[int]
    sf_selected_by_node: list[int]
    snr_by_node: list[float]
    rssi_by_node: list[float]
    mixra_opt_fallback: bool
    timing_s: dict[str, float] | None = None

    @property
    def pdr(self) -> float:
        return packet_delivery_ratio(self.received, self.sent)

    def summary_row(self) -> dict[str, object]:
        return {
            "sent": self.sent,
            "received": self.received,
            "pdr": self.pdr,
            "mean_toa_s": self.mean_toa_s,
            "energy_per_success_bit": self.energy_per_success_bit,
            "snr_db": mean(self.snr_by_node) if self.snr_by_node else 0.0,
            "rssi_dbm": mean(self.rssi_by_node) if self.rssi_by_node else 0.0,
            "mixra_opt_fallback": self.mixra_opt_fallback,
        }


@dataclass(frozen=True)
class NodeQoSCache:
    snr_margins_by_node: list[tuple[float, ...]]
    rssi_margins_by_node: list[tuple[float, ...]]
    qos_margins_by_node: list[float]
    qos_penalties_by_node: list[tuple[float, ...]]
    qos_candidates_by_node: list[tuple[int, ...]]


@dataclass(frozen=True)
class MixraPolicyWeights:
    sf_weight: float
    latency_weight: float
    energy_weight: float
    qos_weight: float


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def _mixra_policy_weights(algorithm: str) -> MixraPolicyWeights:
    if algorithm == "mixra_opt":
        return MixraPolicyWeights(
            sf_weight=0.2,
            latency_weight=0.25,
            energy_weight=0.35,
            qos_weight=0.55,
        )
    return MixraPolicyWeights(
        sf_weight=0.25,
        latency_weight=0.35,
        energy_weight=0.25,
        qos_weight=0.45,
    )


def _precompute_sf_metrics() -> dict[int, tuple[float, float, float]]:
    payload_bytes = DEFAULT_CONFIG.scenario.payload_bytes
    bw_khz = DEFAULT_CONFIG.radio.bandwidth_khz
    cr = coding_rate_to_cr(DEFAULT_CONFIG.radio.coding_rate)
    airtime_by_sf = {
        sf: compute_airtime(payload_bytes=payload_bytes, sf=sf, bw_khz=bw_khz, cr=cr)
        for sf in SF_VALUES
    }
    min_airtime = min(airtime_by_sf.values())
    max_airtime = max(airtime_by_sf.values())
    min_sf = min(SF_VALUES)
    max_sf = max(SF_VALUES)
    return {
        sf: (
            _normalize(sf, min_sf, max_sf),
            _normalize(airtime, min_airtime, max_airtime),
            _normalize(airtime, min_airtime, max_airtime),
        )
        for sf, airtime in airtime_by_sf.items()
    }


SF_METRICS_BY_SF = _precompute_sf_metrics()


def _cluster_mixra_bias(cluster: str | None, clusters: tuple[str, ...]) -> float:
    if not clusters or cluster not in clusters or len(clusters) == 1:
        return 0.0
    index = clusters.index(cluster)
    cluster_scale = (len(clusters) - 1 - index) / (len(clusters) - 1)
    return 2.0 * cluster_scale - 1.0


def _qos_ok(node: NodeLink, sf: int) -> bool:
    index = SF_INDEX[sf]
    return node.snr_margins[index] >= node.qos_margin and node.rssi_margins[index] >= 0.0


def _snr_margin_requirement(snr: float, rssi: float) -> float:
    """Calcule une marge SNR proxy en fonction de la distance/variabilité."""
    rssi_span = 30.0
    distance_factor = min(1.0, max(0.0, (-110.0 - rssi) / rssi_span))
    variability_factor = min(1.0, max(0.0, (-snr) / 20.0))
    return 0.8 + 1.7 * distance_factor + 0.9 * variability_factor


def _density_factor(network_size: int) -> float:
    size_min = min(MIXRA_OPT_BUDGET_BY_SIZE)
    size_max = max(MIXRA_OPT_BUDGET_BY_SIZE)
    if network_size <= size_min:
        return 0.0
    if network_size >= size_max:
        return 1.0
    return (network_size - size_min) / float(size_max - size_min)


def _cluster_radio_adjustment(
    cluster: str | None,
    *,
    clusters: tuple[str, ...],
    density_factor: float,
) -> tuple[float, float, float]:
    """Ajuste les conditions radio selon le cluster QoS."""
    if not cluster or cluster not in clusters or len(clusters) == 1:
        return 0.0, 0.0, 0.0
    index = clusters.index(cluster)
    cluster_scale = (len(clusters) - 1 - index) / (len(clusters) - 1)
    centered = 2.0 * cluster_scale - 1.0
    snr_bonus = (1.6 + 1.4 * density_factor) * centered
    rssi_bonus = (2.0 + 1.2 * density_factor) * centered
    qos_margin_shift = (-0.35 - 0.2 * density_factor) * centered
    return snr_bonus, rssi_bonus, qos_margin_shift


def _adr_smallest_sf(node: NodeLink) -> int:
    """ADR proxy: choisit le plus petit SF satisfaisant les seuils SNR/RSSI."""
    candidates: list[int] = []
    for sf in SF_VALUES:
        if _qos_ok(node, sf):
            candidates.append(sf)
            index = SF_INDEX[sf]
            snr_margin = node.snr_margins[index] - node.qos_margin
            rssi_margin = node.rssi_margins[index]
            min_margin = min(snr_margin, rssi_margin)
            if min_margin >= ADR_MARGIN_BUFFER:
                return sf
    if candidates:
        return candidates[-1]
    return SF_VALUES[-1]


def _mixra_h_assign(
    nodes: Iterable[NodeLink],
    *,
    node_clusters: list[str] | None = None,
    qos_clusters: tuple[str, ...] | None = None,
    policy_weights: MixraPolicyWeights | None = None,
) -> list[int]:
    """MixRA-H proxy: équilibre QoS et pénalise la densité par SF."""
    if qos_clusters is None:
        qos_clusters = tuple(DEFAULT_CONFIG.qos.clusters)
    if policy_weights is None:
        policy_weights = _mixra_policy_weights("mixra_h")
    assignments: list[int] = []
    loads = {sf: 0 for sf in SF_VALUES}
    for node_index, node in enumerate(nodes):
        candidates = [sf for sf in SF_VALUES if _qos_ok(node, sf)]
        if not candidates:
            sf = SF_VALUES[-1]
            assignments.append(sf)
            loads[sf] += 1
            continue

        best_sf = candidates[0]
        best_score = float("inf")
        cluster = (
            node_clusters[node_index]
            if node_clusters is not None and node_index < len(node_clusters)
            else None
        )
        cluster_bias = _cluster_mixra_bias(cluster, qos_clusters)
        for sf in candidates:
            index = SF_INDEX[sf]
            snr_margin = node.snr_margins[index]
            rssi_margin = node.rssi_margins[index]
            qos_margin = min(snr_margin, rssi_margin)
            total_nodes = max(1, len(assignments))
            density = loads[sf] / total_nodes
            load_penalty = loads[sf] + 8.0 * density
            sf_norm, latency_norm, energy_norm = SF_METRICS_BY_SF[sf]
            sf_norm *= 1.0 + MIXRA_CLUSTER_STRENGTH * cluster_bias
            score = (
                load_penalty
                + policy_weights.sf_weight * sf_norm
                + policy_weights.latency_weight * latency_norm
                + policy_weights.energy_weight * energy_norm
                - policy_weights.qos_weight * qos_margin
            )
            if score < best_score:
                best_score = score
                best_sf = sf
        assignments.append(best_sf)
        loads[best_sf] += 1
    return assignments


def _mixra_opt_assign(
    nodes: Iterable[NodeLink],
    *,
    node_clusters: list[str] | None = None,
    qos_clusters: tuple[str, ...] | None = None,
    policy_weights: MixraPolicyWeights | None = None,
    max_iterations: int = 30,
    candidate_subset_size: int = 100,
    convergence_epsilon: float = 1e-3,
    max_evaluations: int = 200,
    subset_seed: int = 0,
    timeout_s: float | None = None,
    allow_fallback: bool = True,
) -> tuple[list[int], bool]:
    """MixRA-Opt proxy: glouton + recherche locale (collisions + QoS)."""
    nodes_list = list(nodes)
    if qos_clusters is None:
        qos_clusters = tuple(DEFAULT_CONFIG.qos.clusters)
    if policy_weights is None:
        policy_weights = _mixra_policy_weights("mixra_opt")
    assignments = _mixra_h_assign(
        nodes_list,
        node_clusters=node_clusters,
        qos_clusters=qos_clusters,
        policy_weights=policy_weights,
    )
    local_rng = random.Random(subset_seed)
    qos_cache = _precompute_node_qos_cache(nodes_list)
    qos_penalties_by_node = qos_cache.qos_penalties_by_node
    qos_candidates_by_node = qos_cache.qos_candidates_by_node
    cluster_biases = [
        _cluster_mixra_bias(cluster, qos_clusters)
        for cluster in (node_clusters or [""] * len(nodes_list))
    ]
    policy_costs_by_node: list[tuple[float, ...]] = []
    for cluster_bias in cluster_biases:
        costs: list[float] = []
        for sf in SF_VALUES:
            sf_norm, latency_norm, energy_norm = SF_METRICS_BY_SF[sf]
            sf_norm *= 1.0 + MIXRA_CLUSTER_STRENGTH * cluster_bias
            cost = (
                policy_weights.sf_weight * sf_norm
                + policy_weights.latency_weight * latency_norm
                + policy_weights.energy_weight * energy_norm
            )
            costs.append(cost)
        policy_costs_by_node.append(tuple(costs))

    def collision_cost(loads: dict[int, int]) -> float:
        return sum(load * load for load in loads.values())

    def _select_candidate_indices() -> list[int]:
        total_nodes = len(nodes_list)
        if total_nodes <= candidate_subset_size:
            return list(range(total_nodes))
        by_sf = {sf: [] for sf in SF_VALUES}
        for idx, sf in enumerate(assignments):
            by_sf[sf].append(idx)
        selected: set[int] = set()
        for sf in SF_VALUES:
            if len(selected) >= candidate_subset_size:
                break
            pool = by_sf[sf]
            if pool:
                selected.add(local_rng.choice(pool))
        remaining = candidate_subset_size - len(selected)
        if remaining <= 0:
            selected_list = list(selected)
            local_rng.shuffle(selected_list)
            return selected_list
        remaining_pool = [idx for idx in range(total_nodes) if idx not in selected]
        if remaining >= len(remaining_pool):
            selected.update(remaining_pool)
        else:
            selected.update(local_rng.sample(remaining_pool, k=remaining))
        selected_list = list(selected)
        local_rng.shuffle(selected_list)
        return selected_list

    loads = {sf: 0 for sf in SF_VALUES}
    for sf in assignments:
        loads[sf] += 1
    current_qos_penalty_by_node = [
        qos_penalties_by_node[idx][SF_INDEX[sf]]
        for idx, sf in enumerate(assignments)
    ]
    current_policy_cost_by_node = [
        policy_costs_by_node[idx][SF_INDEX[sf]]
        for idx, sf in enumerate(assignments)
    ]
    qos_penalty_total = sum(current_qos_penalty_by_node)
    policy_cost_total = sum(current_policy_cost_by_node)
    collision_cost_total = collision_cost(loads)
    qos_weight = 2.5
    current_obj = (
        collision_cost_total
        + qos_weight * qos_penalty_total
        + policy_cost_total
    )

    start_time = perf_counter()
    small_improvement_streak = 0
    evaluations = 0
    def _finalize(
        result: list[int], fallback: bool, *, success: bool
    ) -> tuple[list[int], bool]:
        LOGGER.info("MixRA-Opt executed (evals=%s).", evaluations)
        return result, fallback

    for _ in range(max_iterations):
        improved = False
        candidate_indices = _select_candidate_indices()
        start_obj = current_obj
        for idx in candidate_indices:
            if timeout_s is not None and (perf_counter() - start_time) >= timeout_s:
                LOGGER.warning(
                    "MixRA-Opt timeout atteint (%.2fs).", perf_counter() - start_time
                )
                if allow_fallback:
                    return _finalize(
                        _mixra_h_assign(
                            nodes_list,
                            node_clusters=node_clusters,
                            qos_clusters=qos_clusters,
                            policy_weights=policy_weights,
                        ),
                        True,
                        success=False,
                    )
                return _finalize(assignments, False, success=False)
            evaluations += 1
            if evaluations > max_evaluations:
                if allow_fallback:
                    LOGGER.warning(
                        "MixRA-Opt dépasse le budget (%s > %s évaluations), fallback MixRA-H.",
                        evaluations,
                        max_evaluations,
                    )
                    return _finalize(
                        _mixra_h_assign(
                            nodes_list,
                            node_clusters=node_clusters,
                            qos_clusters=qos_clusters,
                            policy_weights=policy_weights,
                        ),
                        True,
                        success=False,
                    )
                LOGGER.warning(
                    "MixRA-Opt dépasse le budget (%s > %s évaluations), arrêt sans fallback.",
                    evaluations,
                    max_evaluations,
                )
                return _finalize(assignments, False, success=False)
            current_sf = assignments[idx]
            candidates = qos_candidates_by_node[idx]
            if not candidates:
                continue
            best_sf = current_sf
            best_obj = current_obj
            current_load = loads[current_sf]
            current_qos_penalty = current_qos_penalty_by_node[idx]
            current_policy_cost = current_policy_cost_by_node[idx]
            for sf in candidates:
                if sf == current_sf:
                    continue
                candidate_load = loads[sf]
                delta_collision = 2.0 * (candidate_load - current_load + 1.0)
                candidate_qos_penalty = qos_penalties_by_node[idx][SF_INDEX[sf]]
                delta_qos = candidate_qos_penalty - current_qos_penalty
                candidate_policy_cost = policy_costs_by_node[idx][SF_INDEX[sf]]
                delta_policy = candidate_policy_cost - current_policy_cost
                candidate_obj = (
                    current_obj
                    + delta_collision
                    + qos_weight * delta_qos
                    + delta_policy
                )
                if candidate_obj < best_obj:
                    best_obj = candidate_obj
                    best_sf = sf
            if best_sf != current_sf:
                best_load = loads[best_sf]
                loads[current_sf] -= 1
                loads[best_sf] += 1
                assignments[idx] = best_sf
                delta_collision = 2.0 * (best_load - current_load + 1.0)
                new_qos_penalty = qos_penalties_by_node[idx][SF_INDEX[best_sf]]
                qos_penalty_delta = new_qos_penalty - current_qos_penalty
                new_policy_cost = policy_costs_by_node[idx][SF_INDEX[best_sf]]
                policy_cost_delta = new_policy_cost - current_policy_cost
                collision_cost_total += delta_collision
                qos_penalty_total += qos_penalty_delta
                policy_cost_total += policy_cost_delta
                current_qos_penalty_by_node[idx] = new_qos_penalty
                current_policy_cost_by_node[idx] = new_policy_cost
                current_obj = (
                    collision_cost_total
                    + qos_weight * qos_penalty_total
                    + policy_cost_total
                )
                improved = True
        end_obj = current_obj
        improvement = start_obj - end_obj
        if improvement < convergence_epsilon:
            small_improvement_streak += 1
        else:
            small_improvement_streak = 0
        if not improved or small_improvement_streak >= 10:
            break
    return _finalize(assignments, False, success=True)


def mixra_opt_budget_for_size(
    network_size: int,
    *,
    base: int = 0,
    scale: float = 1.0,
) -> int:
    """Retourne un budget d'évaluations MixRA-Opt en fonction de la taille réseau."""
    if network_size <= 0:
        return 0
    if network_size in MIXRA_OPT_BUDGET_BY_SIZE:
        base_budget = MIXRA_OPT_BUDGET_BY_SIZE[network_size]
    else:
        base_budget = int(round(network_size * MIXRA_OPT_BUDGET_PER_NODE))
    return max(0, int(round(base + scale * base_budget)))


def _estimate_received(
    assignments: Iterable[int],
    traffic_times: Iterable[float],
    toa_s_by_node: Iterable[float],
    channels: Iterable[int],
    nodes: Iterable[NodeLink],
    rng: random.Random,
) -> list[bool]:
    """Approxime les collisions en tenant compte des overlaps temps/canal."""
    loads = {sf: 0 for sf in SF_VALUES}
    assignments_list = list(assignments)
    for sf in assignments_list:
        loads[sf] += 1
    capacity_per_sf = 25
    success_prob_by_sf: dict[int, float] = {}
    for sf, load in loads.items():
        if load <= 0:
            success_prob_by_sf[sf] = 0.0
            continue
        if load <= capacity_per_sf:
            delivered = load
        else:
            delivered = capacity_per_sf + (load - capacity_per_sf) * 0.5
        success_prob_by_sf[sf] = max(0.0, min(1.0, delivered / load))
    nodes_list = list(nodes)
    signals = [
        Signal(
            rssi_dbm=node.rssi,
            sf=sf,
            channel_hz=channel,
            start_time_s=start_time,
            end_time_s=start_time + toa_s,
        )
        for sf, start_time, toa_s, channel, node in zip(
            assignments_list,
            traffic_times,
            toa_s_by_node,
            channels,
            nodes_list,
        )
    ]
    sweep_result = compute_co_sf_overlaps(signals)
    overlaps_by_index = sweep_result.overlaps_by_index
    results: list[bool] = []
    for index, signal in enumerate(signals):
        overlap_penalty = 1.0 / (1.0 + len(overlaps_by_index[index]))
        node = nodes_list[index]
        sf_index = SF_INDEX[signal.sf]
        snr_margin = node.snr_margins[sf_index] - node.qos_margin
        rssi_margin = node.rssi_margins[sf_index]
        min_margin = min(snr_margin, rssi_margin)
        quality_factor = 0.2 + 0.8 * max(0.0, min(1.0, (min_margin + 4.0) / 10.0))
        success_probability = (
            success_prob_by_sf[signal.sf] * overlap_penalty * quality_factor
        )
        results.append(rng.random() < success_probability)
    return results


def _generate_nodes(
    count: int,
    seed: int,
    *,
    density_factor: float,
    shadowing_sigma_db: float,
    shadowing_mean_db: float,
    fading_type: str | None,
    fading_sigma_db: float,
    fading_mean_db: float,
    snr_range: tuple[float, float],
    rssi_range: tuple[float, float],
) -> list[NodeLink]:
    rng = random.Random(seed)
    base_tx_power = DEFAULT_CONFIG.radio.tx_power_dbm
    distance_min_km = 0.12
    distance_max_km = 0.8 + 2.6 * density_factor
    distance_mode_km = 0.3 + 0.7 * density_factor
    quality_sigma = 1.1 + 2.0 * density_factor
    tx_power_spread = 1.0 + 1.6 * density_factor
    nodes: list[NodeLink] = []
    for _ in range(count):
        distance_km = rng.triangular(distance_min_km, distance_max_km, distance_mode_km)
        tx_power_dbm = base_tx_power + rng.uniform(-tx_power_spread, tx_power_spread)
        quality_db = rng.gauss(0.0, quality_sigma)
        distance_loss_db = 12.0 * math.log10(1.0 + distance_km * 2.5)
        snr = rng.uniform(*snr_range) + quality_db - 0.6 * distance_loss_db
        rssi = (
            rng.uniform(*rssi_range)
            + quality_db
            - distance_loss_db
            + (tx_power_dbm - base_tx_power)
        )
        shadowing_db = (
            rng.gauss(shadowing_mean_db, shadowing_sigma_db)
            if shadowing_sigma_db > 0
            else shadowing_mean_db
        )
        fading_db = sample_fading_db(
            fading_type,
            sigma_db=fading_sigma_db,
            mean_db=fading_mean_db,
            rng=rng,
        )
        variation_db = shadowing_db + fading_db
        if variation_db != 0.0:
            snr -= variation_db
            rssi -= variation_db
        nodes.append(
            _build_node_link(
                snr,
                rssi,
                distance_km=distance_km,
                tx_power_dbm=tx_power_dbm,
                quality_db=quality_db,
            )
        )
    return nodes


def _build_node_link(
    snr: float,
    rssi: float,
    *,
    distance_km: float,
    tx_power_dbm: float,
    quality_db: float,
    qos_margin_shift: float = 0.0,
) -> NodeLink:
    qos_margin = max(0.1, _snr_margin_requirement(snr, rssi) + qos_margin_shift)
    snr_margins = tuple(snr - SNR_THRESHOLDS[sf] for sf in SF_VALUES)
    rssi_margins = tuple(rssi - RSSI_THRESHOLDS[sf] for sf in SF_VALUES)
    return NodeLink(
        snr=snr,
        rssi=rssi,
        qos_margin=qos_margin,
        snr_margins=snr_margins,
        rssi_margins=rssi_margins,
        distance_km=distance_km,
        tx_power_dbm=tx_power_dbm,
        quality_db=quality_db,
    )


def _apply_cluster_conditions(
    nodes: list[NodeLink],
    node_clusters: list[str],
    *,
    density_factor: float,
    clusters: tuple[str, ...] | None = None,
) -> list[NodeLink]:
    if clusters is None:
        clusters = tuple(DEFAULT_CONFIG.qos.clusters)
    adjusted_nodes: list[NodeLink] = []
    for node, cluster in zip(nodes, node_clusters):
        snr_bonus, rssi_bonus, qos_margin_shift = _cluster_radio_adjustment(
            cluster,
            clusters=clusters,
            density_factor=density_factor,
        )
        adjusted_nodes.append(
            _build_node_link(
                node.snr + snr_bonus,
                node.rssi + rssi_bonus,
                distance_km=node.distance_km,
                tx_power_dbm=node.tx_power_dbm,
                quality_db=node.quality_db + 0.5 * (snr_bonus + rssi_bonus),
                qos_margin_shift=qos_margin_shift,
            )
        )
    return adjusted_nodes


def _precompute_node_qos_cache(nodes_list: list[NodeLink]) -> NodeQoSCache:
    snr_margins_by_node = [node.snr_margins for node in nodes_list]
    rssi_margins_by_node = [node.rssi_margins for node in nodes_list]
    qos_margins_by_node = [node.qos_margin for node in nodes_list]
    qos_penalties_by_node: list[tuple[float, ...]] = []
    qos_candidates_by_node: list[tuple[int, ...]] = []
    for snr_margins, rssi_margins, qos_margin in zip(
        snr_margins_by_node, rssi_margins_by_node, qos_margins_by_node
    ):
        penalties: list[float] = []
        candidates: list[int] = []
        for sf in SF_VALUES:
            index = SF_INDEX[sf]
            snr_margin = snr_margins[index] - qos_margin
            rssi_margin = rssi_margins[index]
            min_margin = min(snr_margin, rssi_margin)
            penalties.append(max(0.0, 2.0 - min_margin))
            if snr_margin >= 0.0 and rssi_margin >= 0.0:
                candidates.append(sf)
        qos_penalties_by_node.append(tuple(penalties))
        qos_candidates_by_node.append(tuple(candidates))
    return NodeQoSCache(
        snr_margins_by_node=snr_margins_by_node,
        rssi_margins_by_node=rssi_margins_by_node,
        qos_margins_by_node=qos_margins_by_node,
        qos_penalties_by_node=qos_penalties_by_node,
        qos_candidates_by_node=qos_candidates_by_node,
    )


def run_simulation(
    sent: int = 120,
    algorithm: str = "adr",
    seed: int = 42,
    *,
    network_size: int | None = None,
    duration_s: float = 3600.0,
    traffic_mode: str = "poisson",
    jitter_range_s: float | None = None,
    mixra_opt_max_iterations: int = 30,
    mixra_opt_candidate_subset_size: int = 100,
    mixra_opt_epsilon: float = 1e-3,
    mixra_opt_max_evaluations: int = 200,
    mixra_opt_budget: int | None = None,
    mixra_opt_budget_base: int = 0,
    mixra_opt_budget_scale: float = 1.0,
    mixra_opt_enabled: bool = True,
    mixra_opt_mode: str = "balanced",
    mixra_opt_timeout_s: float | None = None,
    mixra_opt_no_fallback: bool = False,
    shadowing_sigma_db: float | None = None,
    shadowing_mean_db: float = 0.0,
    fading_type: str | None = "lognormal",
    fading_sigma_db: float = 1.2,
    fading_mean_db: float = 0.0,
    profile_timing: bool = False,
) -> Step1Result:
    """Exécute une simulation minimale.

    Les résultats reposent sur des proxys ADR/MixRA et ne remplacent pas
    l'implémentation complète des algorithmes. Le trafic et le canal
    incluent une variabilité temporelle, et le lien radio applique
    shadowing/fading pour rendre les déclenchements plus fluctuants.
    Le budget MixRA-Opt correspond au nombre maximal d'évaluations, et
    mixra_opt_no_fallback permet d'interdire le basculement MixRA-H.
    """
    rng = random.Random(seed)
    if mixra_opt_timeout_s is None:
        mixra_opt_timeout_s = MIXRA_OPT_EMERGENCY_TIMEOUT_S
    elif mixra_opt_timeout_s <= 0:
        mixra_opt_timeout_s = None
    size_reference = network_size if network_size is not None else sent
    density_factor = _density_factor(int(size_reference))
    if shadowing_sigma_db is None:
        base_shadowing = rng.uniform(5.5, 7.0)
        shadowing_sigma_db = base_shadowing + 2.8 * density_factor
    snr_min = -22.0 - 4.5 * density_factor
    snr_max = 5.0 - 1.8 * density_factor
    rssi_min = -140.0 - 6.0 * density_factor
    rssi_max = -110.0 - 3.0 * density_factor
    jitter_range_value = jitter_range_s
    if jitter_range_value is None:
        base_period = duration_s / max(1, sent)
        jitter_range_value = 0.5 * base_period
    traffic_times = generate_traffic_times(
        sent,
        duration_s=duration_s,
        traffic_mode=traffic_mode,
        jitter_range_s=jitter_range_value,
        rng=rng,
    )
    actual_sent = len(traffic_times)
    nodes = _generate_nodes(
        actual_sent,
        seed,
        density_factor=density_factor,
        shadowing_sigma_db=shadowing_sigma_db,
        shadowing_mean_db=shadowing_mean_db,
        fading_type=fading_type,
        fading_sigma_db=fading_sigma_db,
        fading_mean_db=fading_mean_db,
        snr_range=(snr_min, snr_max),
        rssi_range=(rssi_min, rssi_max),
    )
    cluster_rng = random.Random(seed + 971)
    node_clusters = assign_clusters(actual_sent, rng=cluster_rng)
    nodes = _apply_cluster_conditions(
        nodes,
        node_clusters,
        density_factor=density_factor,
    )
    timings: dict[str, float] | None = {} if profile_timing else None
    start_assignment = perf_counter() if profile_timing else 0.0
    mixra_opt_fallback = False
    if algorithm == "adr":
        assignments = [_adr_smallest_sf(node) for node in nodes]
    elif algorithm == "mixra_h":
        assignments = _mixra_h_assign(nodes, node_clusters=node_clusters)
    elif algorithm == "mixra_opt" and mixra_opt_enabled:
        if mixra_opt_mode not in {"fast", "fast_opt", "full", "balanced"}:
            raise ValueError(
                "mixra_opt_mode doit être 'fast_opt', 'fast', 'balanced' ou 'full' pour l'algorithme mixra_opt."
            )
        allow_fallback = mixra_opt_mode != "full" and not mixra_opt_no_fallback
        if mixra_opt_mode in {"fast", "fast_opt"}:
            mixra_opt_max_iterations = min(mixra_opt_max_iterations, 60)
            mixra_opt_candidate_subset_size = min(mixra_opt_candidate_subset_size, 80)
        if mixra_opt_budget is None:
            computed_budget = mixra_opt_budget_for_size(
                actual_sent,
                base=mixra_opt_budget_base,
                scale=mixra_opt_budget_scale,
            )
            if mixra_opt_mode == "full":
                computed_budget = int(computed_budget * 2.0)
            elif mixra_opt_mode in {"fast", "fast_opt"}:
                computed_budget = max(200, int(computed_budget * 0.4))
            mixra_opt_max_evaluations = max(mixra_opt_max_evaluations, computed_budget)
        else:
            mixra_opt_max_evaluations = mixra_opt_budget
        assignments, mixra_opt_fallback = _mixra_opt_assign(
            nodes,
            node_clusters=node_clusters,
            max_iterations=mixra_opt_max_iterations,
            candidate_subset_size=mixra_opt_candidate_subset_size,
            convergence_epsilon=mixra_opt_epsilon,
            max_evaluations=mixra_opt_max_evaluations,
            subset_seed=seed,
            timeout_s=mixra_opt_timeout_s,
            allow_fallback=allow_fallback,
        )
    elif algorithm == "mixra_opt":
        assignments = _mixra_h_assign(nodes, node_clusters=node_clusters)
        mixra_opt_fallback = True
    else:
        raise ValueError(f"Algorithme inconnu: {algorithm}")
    if profile_timing and timings is not None:
        timings["sf_assignment_s"] = perf_counter() - start_assignment
    payload_bytes = DEFAULT_CONFIG.scenario.payload_bytes
    bw_khz = DEFAULT_CONFIG.radio.bandwidth_khz
    cr = coding_rate_to_cr(DEFAULT_CONFIG.radio.coding_rate)
    airtimes_ms_by_packet = [
        compute_airtime(payload_bytes=payload_bytes, sf=sf, bw_khz=bw_khz, cr=cr)
        for sf in assignments
    ]
    toa_s_by_node = [airtime_ms / 1000.0 for airtime_ms in airtimes_ms_by_packet]
    channels = DEFAULT_CONFIG.radio.channels_hz
    node_channels = [rng.choice(channels) for _ in range(actual_sent)]
    start_interference = perf_counter() if profile_timing else 0.0
    node_received = _estimate_received(
        assignments,
        traffic_times,
        toa_s_by_node,
        node_channels,
        nodes,
        rng,
    )
    if profile_timing and timings is not None:
        timings["interference_s"] = perf_counter() - start_interference
    received = sum(1 for value in node_received if value)
    mean_toa = mean_toa_s(airtimes_ms_by_packet)
    payload_bits_success = payload_bytes * 8 * received
    energy_per_bit = energy_per_success_bit(
        airtimes_ms_by_packet, payload_bits_success, DEFAULT_CONFIG.radio.tx_power_dbm
    )
    packet_ids = list(range(actual_sent))
    return Step1Result(
        sent=actual_sent,
        received=received,
        energy_per_success_bit=energy_per_bit,
        mean_toa_s=mean_toa,
        node_clusters=node_clusters,
        node_received=node_received,
        toa_s_by_node=toa_s_by_node,
        packet_ids=packet_ids,
        sf_selected_by_node=list(assignments),
        snr_by_node=[node.snr for node in nodes],
        rssi_by_node=[node.rssi for node in nodes],
        mixra_opt_fallback=mixra_opt_fallback,
        timing_s=timings,
    )
