"""Simulation de l'étape 1 avec proxys ADR/MixRA.

Les implémentations ci-dessous fournissent des heuristiques simples et
reproductibles pour ADR, MixRA-H et MixRA-Opt. Elles servent de substituts
cohérents lorsque les formules exactes ne sont pas disponibles.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
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

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeLink:
    snr: float
    rssi: float
    qos_margin: float
    snr_margins: tuple[float, ...]
    rssi_margins: tuple[float, ...]


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
            "mixra_opt_fallback": self.mixra_opt_fallback,
        }


def _qos_ok(node: NodeLink, sf: int) -> bool:
    index = SF_INDEX[sf]
    return node.snr_margins[index] >= node.qos_margin and node.rssi_margins[index] >= 0.0


def _snr_margin_requirement(snr: float, rssi: float) -> float:
    """Calcule une marge SNR proxy en fonction de la distance/variabilité."""
    rssi_span = 30.0
    distance_factor = min(1.0, max(0.0, (-110.0 - rssi) / rssi_span))
    variability_factor = min(1.0, max(0.0, (-snr) / 20.0))
    return 0.8 + 1.7 * distance_factor + 0.9 * variability_factor


def _adr_smallest_sf(node: NodeLink) -> int:
    """ADR proxy: choisit le plus petit SF satisfaisant les seuils SNR/RSSI."""
    for sf in SF_VALUES:
        if _qos_ok(node, sf):
            return sf
    return SF_VALUES[-1]


def _mixra_h_assign(nodes: Iterable[NodeLink]) -> list[int]:
    """MixRA-H proxy: équilibre QoS et pénalise la densité par SF."""
    assignments: list[int] = []
    loads = {sf: 0 for sf in SF_VALUES}
    for node in nodes:
        candidates = [sf for sf in SF_VALUES if _qos_ok(node, sf)]
        if not candidates:
            sf = SF_VALUES[-1]
            assignments.append(sf)
            loads[sf] += 1
            continue

        best_sf = candidates[0]
        best_score = float("inf")
        for sf in candidates:
            index = SF_INDEX[sf]
            snr_margin = node.snr_margins[index]
            rssi_margin = node.rssi_margins[index]
            qos_margin = min(snr_margin, rssi_margin)
            total_nodes = max(1, len(assignments))
            density = loads[sf] / total_nodes
            load_penalty = loads[sf] + 8.0 * density
            score = load_penalty - 0.35 * qos_margin
            if score < best_score:
                best_score = score
                best_sf = sf
        assignments.append(best_sf)
        loads[best_sf] += 1
    return assignments


def _mixra_opt_assign(
    nodes: Iterable[NodeLink],
    *,
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
    assignments = _mixra_h_assign(nodes_list)
    local_rng = random.Random(subset_seed)
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
    qos_penalty_total = sum(current_qos_penalty_by_node)
    collision_cost_total = collision_cost(loads)
    current_obj = collision_cost_total + 2.5 * qos_penalty_total

    start_time = perf_counter()
    small_improvement_streak = 0
    evaluations = 0
    def _finalize(
        result: list[int], fallback: bool, *, success: bool
    ) -> tuple[list[int], bool]:
        if success:
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
                    return _finalize(_mixra_h_assign(nodes_list), True, success=False)
                return _finalize(assignments, False, success=False)
            evaluations += 1
            if evaluations > max_evaluations:
                if allow_fallback:
                    LOGGER.warning(
                        "MixRA-Opt dépasse le budget (%s > %s évaluations), fallback MixRA-H.",
                        evaluations,
                        max_evaluations,
                    )
                    return _finalize(_mixra_h_assign(nodes_list), True, success=False)
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
            for sf in candidates:
                if sf == current_sf:
                    continue
                candidate_load = loads[sf]
                delta_collision = 2.0 * (candidate_load - current_load + 1.0)
                candidate_qos_penalty = qos_penalties_by_node[idx][SF_INDEX[sf]]
                delta_qos = candidate_qos_penalty - current_qos_penalty
                candidate_obj = (
                    collision_cost_total
                    + delta_collision
                    + 2.5 * (qos_penalty_total + delta_qos)
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
                collision_cost_total += delta_collision
                qos_penalty_total += qos_penalty_delta
                current_qos_penalty_by_node[idx] = new_qos_penalty
                current_obj = collision_cost_total + 2.5 * qos_penalty_total
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
    rssis: Iterable[float],
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
    signals = [
        Signal(
            rssi_dbm=rssi,
            sf=sf,
            channel_hz=channel,
            start_time_s=start_time,
            end_time_s=start_time + toa_s,
        )
        for sf, start_time, toa_s, channel, rssi in zip(
            assignments_list,
            traffic_times,
            toa_s_by_node,
            channels,
            rssis,
        )
    ]
    sweep_result = compute_co_sf_overlaps(signals)
    overlaps_by_index = sweep_result.overlaps_by_index
    results: list[bool] = []
    for index, signal in enumerate(signals):
        overlap_penalty = 1.0 / (1.0 + len(overlaps_by_index[index]))
        success_probability = success_prob_by_sf[signal.sf] * overlap_penalty
        results.append(rng.random() < success_probability)
    return results


def _generate_nodes(
    count: int,
    seed: int,
    *,
    shadowing_sigma_db: float,
    shadowing_mean_db: float,
    fading_type: str | None,
    fading_sigma_db: float,
    fading_mean_db: float,
) -> list[NodeLink]:
    rng = random.Random(seed)
    nodes: list[NodeLink] = []
    for _ in range(count):
        snr = rng.uniform(-22.0, 5.0)
        rssi = rng.uniform(-140.0, -110.0)
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
        qos_margin = _snr_margin_requirement(snr, rssi)
        snr_margins = tuple(snr - SNR_THRESHOLDS[sf] for sf in SF_VALUES)
        rssi_margins = tuple(rssi - RSSI_THRESHOLDS[sf] for sf in SF_VALUES)
        nodes.append(
            NodeLink(
                snr=snr,
                rssi=rssi,
                qos_margin=qos_margin,
                snr_margins=snr_margins,
                rssi_margins=rssi_margins,
            )
        )
    return nodes


def run_simulation(
    sent: int = 120,
    algorithm: str = "adr",
    seed: int = 42,
    *,
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
    shadowing_sigma_db: float = 7.0,
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
    Le budget MixRA-Opt correspond au nombre maximal d'évaluations.
    """
    rng = random.Random(seed)
    if mixra_opt_timeout_s is None:
        mixra_opt_timeout_s = MIXRA_OPT_EMERGENCY_TIMEOUT_S
    elif mixra_opt_timeout_s <= 0:
        mixra_opt_timeout_s = None
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
        shadowing_sigma_db=shadowing_sigma_db,
        shadowing_mean_db=shadowing_mean_db,
        fading_type=fading_type,
        fading_sigma_db=fading_sigma_db,
        fading_mean_db=fading_mean_db,
    )
    timings: dict[str, float] | None = {} if profile_timing else None
    start_assignment = perf_counter() if profile_timing else 0.0
    mixra_opt_fallback = False
    if algorithm == "adr":
        assignments = [_adr_smallest_sf(node) for node in nodes]
    elif algorithm == "mixra_h":
        assignments = _mixra_h_assign(nodes)
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
            max_iterations=mixra_opt_max_iterations,
            candidate_subset_size=mixra_opt_candidate_subset_size,
            convergence_epsilon=mixra_opt_epsilon,
            max_evaluations=mixra_opt_max_evaluations,
            subset_seed=seed,
            timeout_s=mixra_opt_timeout_s,
            allow_fallback=allow_fallback,
        )
    elif algorithm == "mixra_opt":
        assignments = _mixra_h_assign(nodes)
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
        [node.rssi for node in nodes],
        rng,
    )
    if profile_timing and timings is not None:
        timings["interference_s"] = perf_counter() - start_interference
    node_clusters = assign_clusters(actual_sent, rng=rng)
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
        mixra_opt_fallback=mixra_opt_fallback,
        timing_s=timings,
    )
