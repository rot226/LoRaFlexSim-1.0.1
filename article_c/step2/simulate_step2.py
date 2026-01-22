"""Simulation de l'étape 2 (proxy UCB1 et comparaisons)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Literal

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_rows, write_simulation_results
from article_c.common.lora_phy import bitrate_lora, coding_rate_to_cr, compute_airtime
from article_c.common.utils import assign_clusters, generate_traffic_times
from article_c.step2.bandit_ucb1 import BanditUCB1


SF_VALUES = (7, 8, 9, 10, 11, 12)


@dataclass(frozen=True)
class WindowMetrics:
    success_rate: float
    bitrate_norm: float
    energy_norm: float
    collision_norm: float


@dataclass(frozen=True)
class AlgoRewardWeights:
    sf_weight: float
    latency_weight: float
    energy_weight: float
    collision_weight: float
    exploration_floor: float = 0.0


@dataclass(frozen=True)
class Step2Result:
    raw_rows: list[dict[str, object]]
    selection_prob_rows: list[dict[str, object]]
    learning_curve_rows: list[dict[str, object]]


logger = logging.getLogger(__name__)


def _clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _compute_reward(
    success_rate: float,
    sf_norm: float,
    latency_norm: float,
    energy_norm: float,
    collision_norm: float,
    weights: AlgoRewardWeights,
    lambda_energy: float,
    lambda_collision: float,
) -> float:
    energy_weight = weights.energy_weight * (1.0 + lambda_energy)
    total_weight = weights.sf_weight + weights.latency_weight + energy_weight
    if total_weight <= 0:
        total_weight = 1.0
    sf_score = 1.0 - sf_norm
    latency_score = 1.0 - latency_norm
    energy_score = 1.0 - energy_norm
    weighted_quality = (
        weights.sf_weight * sf_score
        + weights.latency_weight * latency_score
        + energy_weight * energy_score
    ) / total_weight
    reward = success_rate * weighted_quality - (
        lambda_collision * weights.collision_weight * collision_norm
    )
    return _clip(reward, 0.0, 1.0)


def _clamp_range(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _network_load_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    return max(0.6, ratio**0.6)


def _traffic_coeff_size_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    return max(0.7, ratio**0.35)


def _cluster_traffic_factor(cluster: str, clusters: tuple[str, ...]) -> float:
    if not clusters:
        return 1.0
    if cluster not in clusters:
        return 1.0
    index = clusters.index(cluster)
    if len(clusters) == 1:
        return 1.0
    max_factor = 1.35
    min_factor = 0.75
    step = (max_factor - min_factor) / (len(clusters) - 1)
    return max_factor - step * index


def _cluster_shadowing_sigma_factor(cluster: str, clusters: tuple[str, ...]) -> float:
    if not clusters:
        return 1.0
    if cluster not in clusters:
        return 1.0
    index = clusters.index(cluster)
    if len(clusters) == 1:
        return 1.0
    min_factor = 0.85
    max_factor = 1.2
    step = (max_factor - min_factor) / (len(clusters) - 1)
    return min_factor + step * index


def _mixra_cluster_qos_factor(cluster: str, clusters: tuple[str, ...]) -> float:
    if not clusters or cluster not in clusters or len(clusters) == 1:
        return 1.0
    index = clusters.index(cluster)
    cluster_scale = (len(clusters) - 1 - index) / (len(clusters) - 1)
    return 0.85 + 0.35 * (1.0 - cluster_scale)


def _congestion_collision_probability(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 0.0
    overload = max(0.0, (network_size / reference_size) - 1.0)
    return _clip(0.12 * overload, 0.0, 0.45)


def _compute_window_metrics(
    successes: int,
    traffic_sent: int,
    bitrate_norm: float,
    energy_norm: float,
    collision_norm: float,
) -> WindowMetrics:
    success_rate = successes / traffic_sent if traffic_sent > 0 else 0.0
    return WindowMetrics(
        success_rate=success_rate,
        bitrate_norm=bitrate_norm,
        energy_norm=energy_norm,
        collision_norm=collision_norm,
    )


def _compute_collision_successes(
    transmissions: dict[int, list[tuple[float, float, int]]],
    *,
    rng: random.Random | None = None,
    approx_threshold: int = 5000,
    approx_sample_size: int = 2500,
) -> tuple[dict[int, int], int, bool]:
    def _compute_collisions(
        events: list[tuple[float, float, int]]
    ) -> list[bool]:
        collided = [False] * len(events)
        indexed_events = sorted(enumerate(events), key=lambda item: item[1][0])
        group_indices: list[int] = []
        group_end: float | None = None
        for event_index, (start, end, _node_id) in indexed_events:
            if group_end is None or start >= group_end:
                if len(group_indices) > 1:
                    for index in group_indices:
                        collided[index] = True
                group_indices = [event_index]
                group_end = end
            else:
                group_indices.append(event_index)
                if end > group_end:
                    group_end = end
        if len(group_indices) > 1:
            for index in group_indices:
                collided[index] = True
        return collided

    total_transmissions = sum(len(events) for events in transmissions.values())
    approx_mode = total_transmissions > approx_threshold
    rng = rng or random.Random(0)
    successes_by_node: dict[int, int] = {}
    for events in transmissions.values():
        if not events:
            continue
        per_node_total: dict[int, int] = {}
        for _start, _end, node_id in events:
            per_node_total[node_id] = per_node_total.get(node_id, 0) + 1
        sample_probability = 1.0
        sampled_events = events
        if approx_mode and len(events) > approx_sample_size:
            sample_probability = approx_sample_size / len(events)
            sampled_events = [
                event for event in events if rng.random() < sample_probability
            ]
            if not sampled_events:
                sampled_events = [rng.choice(events)]
        collided = _compute_collisions(sampled_events)
        sample_successes: dict[int, int] = {}
        for event_index, (_start, _end, node_id) in enumerate(sampled_events):
            if not collided[event_index]:
                sample_successes[node_id] = sample_successes.get(node_id, 0) + 1
        if sample_probability < 1.0:
            for node_id, successes in sample_successes.items():
                estimated = int(round(successes / sample_probability))
                successes_by_node[node_id] = successes_by_node.get(node_id, 0) + min(
                    estimated, per_node_total[node_id]
                )
        else:
            for node_id, successes in sample_successes.items():
                successes_by_node[node_id] = (
                    successes_by_node.get(node_id, 0) + successes
                )
    return successes_by_node, total_transmissions, approx_mode


def _sample_log_normal_shadowing(
    rng: random.Random, mean_db: float, sigma_db: float
) -> tuple[float, float]:
    shadowing_db = rng.gauss(mean_db, sigma_db)
    return shadowing_db, 10 ** (-shadowing_db / 10.0)


def _weights_for_algo(algorithm: str, n_arms: int) -> list[float]:
    if algorithm == "mixra_h":
        base = [0.28, 0.24, 0.2, 0.15, 0.09, 0.04]
    elif algorithm == "mixra_opt":
        base = [0.18, 0.2, 0.2, 0.17, 0.15, 0.1]
    else:
        base = [1.0] + [0.0] * 5
    weights = base[:n_arms]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def _reward_weights_for_algo(algorithm: str) -> AlgoRewardWeights:
    if algorithm == "adr":
        return AlgoRewardWeights(
            sf_weight=0.5,
            latency_weight=0.3,
            energy_weight=0.2,
            collision_weight=0.25,
        )
    if algorithm == "mixra_h":
        return AlgoRewardWeights(
            sf_weight=0.3,
            latency_weight=0.25,
            energy_weight=0.45,
            collision_weight=0.3,
        )
    if algorithm == "mixra_opt":
        return AlgoRewardWeights(
            sf_weight=0.25,
            latency_weight=0.2,
            energy_weight=0.55,
            collision_weight=0.35,
        )
    return AlgoRewardWeights(
        sf_weight=0.4,
        latency_weight=0.3,
        energy_weight=0.3,
        collision_weight=0.25,
        exploration_floor=0.08,
    )


def _apply_cluster_bias(
    weights: list[float], cluster: str, clusters: tuple[str, ...], strength: float
) -> list[float]:
    if not clusters or cluster not in clusters or len(weights) <= 1:
        return weights
    index = clusters.index(cluster)
    if len(clusters) == 1:
        return weights
    cluster_scale = (len(clusters) - 1 - index) / (len(clusters) - 1)
    cluster_bias = 2.0 * cluster_scale - 1.0
    ramp = [
        2.0 * (arm_index / (len(weights) - 1)) - 1.0 for arm_index in range(len(weights))
    ]
    adjusted = [
        max(0.05, weight * (1.0 + strength * cluster_bias * ramp_value))
        for weight, ramp_value in zip(weights, ramp)
    ]
    total = sum(adjusted) or 1.0
    return [value / total for value in adjusted]


def _select_adr_arm(
    link_quality: float, sf_values: list[int], cluster: str, clusters: tuple[str, ...]
) -> int:
    if len(sf_values) <= 1:
        return 0
    cluster_scale = 0.5
    if clusters and cluster in clusters and len(clusters) > 1:
        cluster_scale = (len(clusters) - 1 - clusters.index(cluster)) / (
            len(clusters) - 1
        )
    target_quality = 0.55 + 0.25 * cluster_scale
    normalized_gap = max(0.0, target_quality - link_quality) / max(target_quality, 1e-6)
    arm_index = int(round(normalized_gap * (len(sf_values) - 1)))
    return max(0, min(len(sf_values) - 1, arm_index))


def _algo_label(algorithm: str) -> str:
    return {
        "adr": "ADR",
        "mixra_h": "MixRA-H",
        "mixra_opt": "MixRA-Opt",
        "ucb1_sf": "UCB1-SF",
    }.get(algorithm, algorithm)


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def run_simulation(
    algorithm: Literal["adr", "mixra_h", "mixra_opt", "ucb1_sf"] = "ucb1_sf",
    n_rounds: int = 20,
    n_nodes: int = 12,
    n_arms: int | None = None,
    window_size: int = DEFAULT_CONFIG.rl.window_w,
    lambda_energy: float = DEFAULT_CONFIG.rl.lambda_energy,
    epsilon_greedy: float = 0.03,
    density: float | None = None,
    snir_mode: str = "snir_on",
    seed: int = 42,
    traffic_mode: str | None = None,
    jitter_range_s: float | None = None,
    window_duration_s: float | None = None,
    traffic_coeff_min: float | None = None,
    traffic_coeff_max: float | None = None,
    traffic_coeff_enabled: bool | None = None,
    window_delay_enabled: bool | None = None,
    window_delay_range_s: float | None = None,
    shadowing_sigma_db: float | None = None,
    output_dir: Path | None = None,
) -> Step2Result:
    """Exécute une simulation proxy de l'étape 2."""
    rng = random.Random(seed)
    step2_defaults = DEFAULT_CONFIG.step2
    traffic_mode_value = "poisson" if traffic_mode is None else traffic_mode
    jitter_range_value = jitter_range_s
    window_duration_value = (
        step2_defaults.window_duration_s if window_duration_s is None else window_duration_s
    )
    traffic_coeff_min_value = (
        step2_defaults.traffic_coeff_min if traffic_coeff_min is None else traffic_coeff_min
    )
    traffic_coeff_max_value = (
        step2_defaults.traffic_coeff_max if traffic_coeff_max is None else traffic_coeff_max
    )
    if traffic_coeff_min_value > traffic_coeff_max_value:
        traffic_coeff_min_value, traffic_coeff_max_value = (
            traffic_coeff_max_value,
            traffic_coeff_min_value,
        )
    traffic_coeff_enabled_value = (
        step2_defaults.traffic_coeff_enabled
        if traffic_coeff_enabled is None
        else traffic_coeff_enabled
    )
    window_delay_enabled_value = (
        step2_defaults.window_delay_enabled
        if window_delay_enabled is None
        else window_delay_enabled
    )
    window_delay_range_value = (
        step2_defaults.window_delay_range_s
        if window_delay_range_s is None
        else window_delay_range_s
    )
    epsilon_greedy = _clip(epsilon_greedy, 0.0, 1.0)
    if density is not None:
        n_nodes = int(density)
    network_size_value = n_nodes
    if n_nodes <= 0:
        if n_nodes == 0:
            logger.error("network_size == 0 avant écriture des résultats.")
        raise ValueError("network_size doit être strictement positif.")
    algo_label = _algo_label(algorithm)
    raw_rows: list[dict[str, object]] = []
    selection_prob_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    node_clusters = assign_clusters(n_nodes, rng=rng)
    reference_size = max(1, DEFAULT_CONFIG.rl.window_w)
    load_factor = _network_load_factor(n_nodes, reference_size)
    congestion_probability = _congestion_collision_probability(n_nodes, reference_size)
    qos_clusters = tuple(DEFAULT_CONFIG.qos.clusters)
    traffic_size_factor = _traffic_coeff_size_factor(n_nodes, reference_size)
    traffic_coeffs = [
        _clamp_range(
            (
                rng.uniform(traffic_coeff_min_value, traffic_coeff_max_value)
                if traffic_coeff_enabled_value
                else 1.0
            )
            * (
                traffic_size_factor
                * _cluster_traffic_factor(node_clusters[node_id], qos_clusters)
            ),
            0.4,
            2.5,
        )
        for node_id in range(n_nodes)
    ]
    base_rate_multipliers = [rng.uniform(0.7, 1.3) for _ in range(n_nodes)]

    sf_values = list(SF_VALUES)
    if n_arms is None:
        n_arms = len(sf_values)
    sf_values = sf_values[:n_arms]
    payload_bytes = DEFAULT_CONFIG.scenario.payload_bytes
    bw_khz = DEFAULT_CONFIG.radio.bandwidth_khz
    cr_value = coding_rate_to_cr(DEFAULT_CONFIG.radio.coding_rate)

    airtime_by_sf = {
        sf: compute_airtime(payload_bytes=payload_bytes, sf=sf, bw_khz=bw_khz, cr=cr_value)
        for sf in sf_values
    }
    bitrate_by_sf = {
        sf: bitrate_lora(sf=sf, bw=bw_khz, cr=cr_value) for sf in sf_values
    }
    min_bitrate = min(bitrate_by_sf.values())
    max_bitrate = max(bitrate_by_sf.values())
    min_airtime = min(airtime_by_sf.values())
    max_airtime = max(airtime_by_sf.values())
    bitrate_norm_by_sf = {
        sf: _normalize(bitrate, min_bitrate, max_bitrate)
        for sf, bitrate in bitrate_by_sf.items()
    }
    energy_norm_by_sf = {
        sf: _normalize(airtime, min_airtime, max_airtime)
        for sf, airtime in airtime_by_sf.items()
    }
    shadowing_mean_db = DEFAULT_CONFIG.scenario.shadowing_mean_db
    base_shadowing_sigma_db = (
        rng.uniform(6.0, 8.0) if shadowing_sigma_db is None else shadowing_sigma_db
    )
    lambda_collision = _clip(0.15 + 0.45 * lambda_energy, 0.08, 0.7)
    reward_weights = _reward_weights_for_algo(algorithm)
    sf_norm_by_sf = {
        sf: _normalize(sf, min(sf_values), max(sf_values)) for sf in sf_values
    }
    latency_norm_by_sf = energy_norm_by_sf

    if algorithm == "ucb1_sf":
        bandit = BanditUCB1(
            n_arms=n_arms,
            warmup_rounds=DEFAULT_CONFIG.rl.warmup,
            epsilon_min=0.02,
        )
        exploration_epsilon = max(epsilon_greedy, reward_weights.exploration_floor)
        window_start_s = 0.0
        for round_id in range(n_rounds):
            arm_index = bandit.select_arm()
            if exploration_epsilon > 0.0 and rng.random() < exploration_epsilon:
                arm_index = rng.randrange(n_arms)
            window_rewards: list[float] = []
            if round_id > 0:
                delay_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value
                    else 0.0
                )
                window_start_s += window_duration_value + delay_s
            node_windows: list[dict[str, object]] = []
            for node_id in range(n_nodes):
                sf_value = sf_values[arm_index]
                rate_multiplier = base_rate_multipliers[node_id]
                cluster = node_clusters[node_id]
                expected_sent = max(
                    1,
                    int(
                        round(
                            window_size
                            * traffic_coeffs[node_id]
                            * rate_multiplier
                            * load_factor
                        )
                    ),
                )
                base_period_s = window_duration_value / expected_sent
                jitter_range_node_s = (
                    0.5 * base_period_s
                    if jitter_range_value is None
                    else jitter_range_value
                )
                traffic_times = generate_traffic_times(
                    expected_sent,
                    duration_s=window_duration_value,
                    traffic_mode=traffic_mode_value,
                    jitter_range_s=jitter_range_node_s,
                    rng=rng,
                )
                shadowing_sigma_db_node = _clamp_range(
                    base_shadowing_sigma_db
                    * _cluster_shadowing_sigma_factor(cluster, qos_clusters),
                    2.5,
                    12.0,
                )
                shadowing_db, shadowing_linear = _sample_log_normal_shadowing(
                    rng,
                    mean_db=shadowing_mean_db,
                    sigma_db=shadowing_sigma_db_node,
                )
                link_quality = _clip(shadowing_linear, 0.0, 1.0)
                node_offset_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value and window_delay_range_value > 0
                    else 0.0
                )
                tx_starts = [window_start_s + node_offset_s + t for t in traffic_times]
                node_windows.append(
                    {
                        "node_id": node_id,
                        "arm_index": arm_index,
                        "sf": sf_value,
                        "node_offset_s": node_offset_s,
                        "traffic_coeff": traffic_coeffs[node_id],
                        "rate_multiplier": rate_multiplier,
                        "traffic_sent": len(traffic_times),
                        "tx_starts": tx_starts,
                        "shadowing_db": shadowing_db,
                        "shadowing_sigma_db": shadowing_sigma_db_node,
                        "link_quality": link_quality,
                    }
                )
            transmissions_by_sf: dict[int, list[tuple[float, float, int]]] = {}
            for node_window in node_windows:
                sf_value = int(node_window["sf"])
                airtime = airtime_by_sf[sf_value]
                for start_time in node_window["tx_starts"]:
                    transmissions_by_sf.setdefault(sf_value, []).append(
                        (start_time, start_time + airtime, int(node_window["node_id"]))
                    )
            (
                successes_by_node,
                transmission_count,
                approx_collision_mode,
            ) = _compute_collision_successes(transmissions_by_sf, rng=rng)
            if approx_collision_mode:
                logger.debug(
                    "Mode approx collisions activé (%s transmissions).",
                    transmission_count,
                )
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = int(node_window["traffic_sent"])
                successes = successes_by_node.get(node_id, 0)
                if successes > 0 and congestion_probability > 0.0:
                    successes = sum(
                        1
                        for _ in range(successes)
                        if rng.random() > congestion_probability
                    )
                link_quality = float(node_window["link_quality"])
                if link_quality < 1.0 and successes > 0:
                    successes = sum(
                        1 for _ in range(successes) if rng.random() < link_quality
                    )
                airtime_norm = energy_norm_by_sf[sf_value]
                collision_norm = _clip(
                    airtime_norm
                    * (1.0 + congestion_probability)
                    * (1.0 - (successes / traffic_sent if traffic_sent > 0 else 0.0)),
                    0.0,
                    1.0,
                )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    airtime_norm,
                    collision_norm,
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    sf_norm_by_sf[sf_value],
                    latency_norm_by_sf[sf_value],
                    metrics.energy_norm,
                    metrics.collision_norm,
                    reward_weights,
                    lambda_energy,
                    lambda_collision,
                )
                window_rewards.append(reward)
                raw_rows.append(
                    {
                        "network_size": network_size_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "cluster": node_clusters[node_id],
                        "round": round_id,
                        "node_id": node_id,
                        "sf": sf_value,
                        "window_start_s": window_start_s,
                        "node_offset_s": node_window["node_offset_s"],
                        "traffic_coeff": node_window["traffic_coeff"],
                        "rate_multiplier": node_window["rate_multiplier"],
                        "traffic_sent": traffic_sent,
                        "shadowing_db": node_window["shadowing_db"],
                        "shadowing_sigma_db": node_window["shadowing_sigma_db"],
                        "link_quality": node_window["link_quality"],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "collision_norm": metrics.collision_norm,
                        "reward": reward,
                    }
                )
                raw_rows.append(
                    {
                        "network_size": network_size_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "cluster": "all",
                        "round": round_id,
                        "node_id": node_id,
                        "sf": sf_value,
                        "window_start_s": window_start_s,
                        "node_offset_s": node_window["node_offset_s"],
                        "traffic_coeff": node_window["traffic_coeff"],
                        "rate_multiplier": node_window["rate_multiplier"],
                        "traffic_sent": traffic_sent,
                        "shadowing_db": node_window["shadowing_db"],
                        "shadowing_sigma_db": node_window["shadowing_sigma_db"],
                        "link_quality": node_window["link_quality"],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "collision_norm": metrics.collision_norm,
                        "reward": reward,
                    }
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
            logger.info(
                "Round %s - %s : récompense moyenne = %.4f",
                round_id,
                algo_label,
                avg_reward,
            )
            learning_curve_rows.append(
                {
                    "network_size": network_size_value,
                    "round": round_id,
                    "algo": algo_label,
                    "avg_reward": avg_reward,
                }
            )
            bandit.update(arm_index, avg_reward)
            total = sum(bandit.counts) or 1
            for sf_index, sf_value in enumerate(sf_values):
                selection_prob_rows.append(
                    {
                        "network_size": network_size_value,
                        "round": round_id,
                        "sf": sf_value,
                        "selection_prob": bandit.counts[sf_index] / total,
                    }
                )
    elif algorithm in {"adr", "mixra_h", "mixra_opt"}:
        weights = _weights_for_algo(algorithm, n_arms)
        mixra_strength = 0.45 if algorithm == "mixra_h" else 0.25
        window_start_s = 0.0
        for round_id in range(n_rounds):
            window_rewards: list[float] = []
            if round_id > 0:
                delay_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value
                    else 0.0
                )
                window_start_s += window_duration_value + delay_s
            node_windows: list[dict[str, object]] = []
            for node_id in range(n_nodes):
                rate_multiplier = base_rate_multipliers[node_id]
                cluster = node_clusters[node_id]
                expected_sent = max(
                    1,
                    int(
                        round(
                            window_size
                            * traffic_coeffs[node_id]
                            * rate_multiplier
                            * load_factor
                        )
                    ),
                )
                base_period_s = window_duration_value / expected_sent
                jitter_range_node_s = (
                    0.5 * base_period_s
                    if jitter_range_value is None
                    else jitter_range_value
                )
                traffic_times = generate_traffic_times(
                    expected_sent,
                    duration_s=window_duration_value,
                    traffic_mode=traffic_mode_value,
                    jitter_range_s=jitter_range_node_s,
                    rng=rng,
                )
                shadowing_sigma_db_node = _clamp_range(
                    base_shadowing_sigma_db
                    * _cluster_shadowing_sigma_factor(cluster, qos_clusters),
                    2.5,
                    12.0,
                )
                shadowing_db, shadowing_linear = _sample_log_normal_shadowing(
                    rng,
                    mean_db=shadowing_mean_db,
                    sigma_db=shadowing_sigma_db_node,
                )
                link_quality = _clip(shadowing_linear, 0.0, 1.0)
                if algorithm == "adr":
                    arm_index = _select_adr_arm(
                        link_quality, sf_values, cluster, qos_clusters
                    )
                else:
                    cluster_qos_factor = _mixra_cluster_qos_factor(
                        cluster, qos_clusters
                    )
                    cluster_weights = _apply_cluster_bias(
                        weights,
                        cluster,
                        qos_clusters,
                        mixra_strength * cluster_qos_factor,
                    )
                    arm_index = rng.choices(range(n_arms), weights=cluster_weights, k=1)[0]
                sf_value = sf_values[arm_index]
                node_offset_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value and window_delay_range_value > 0
                    else 0.0
                )
                tx_starts = [window_start_s + node_offset_s + t for t in traffic_times]
                node_windows.append(
                    {
                        "node_id": node_id,
                        "arm_index": arm_index,
                        "sf": sf_value,
                        "node_offset_s": node_offset_s,
                        "traffic_coeff": traffic_coeffs[node_id],
                        "rate_multiplier": rate_multiplier,
                        "traffic_sent": len(traffic_times),
                        "tx_starts": tx_starts,
                        "shadowing_db": shadowing_db,
                        "shadowing_sigma_db": shadowing_sigma_db_node,
                        "link_quality": link_quality,
                    }
                )
            transmissions_by_sf: dict[int, list[tuple[float, float, int]]] = {}
            for node_window in node_windows:
                sf_value = int(node_window["sf"])
                airtime = airtime_by_sf[sf_value]
                for start_time in node_window["tx_starts"]:
                    transmissions_by_sf.setdefault(sf_value, []).append(
                        (start_time, start_time + airtime, int(node_window["node_id"]))
                    )
            (
                successes_by_node,
                transmission_count,
                approx_collision_mode,
            ) = _compute_collision_successes(transmissions_by_sf, rng=rng)
            if approx_collision_mode:
                logger.debug(
                    "Mode approx collisions activé (%s transmissions).",
                    transmission_count,
                )
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = int(node_window["traffic_sent"])
                successes = successes_by_node.get(node_id, 0)
                if successes > 0 and congestion_probability > 0.0:
                    successes = sum(
                        1
                        for _ in range(successes)
                        if rng.random() > congestion_probability
                    )
                link_quality = float(node_window["link_quality"])
                if link_quality < 1.0 and successes > 0:
                    successes = sum(
                        1 for _ in range(successes) if rng.random() < link_quality
                    )
                airtime_norm = energy_norm_by_sf[sf_value]
                collision_norm = _clip(
                    airtime_norm
                    * (1.0 + congestion_probability)
                    * (1.0 - (successes / traffic_sent if traffic_sent > 0 else 0.0)),
                    0.0,
                    1.0,
                )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    airtime_norm,
                    collision_norm,
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    sf_norm_by_sf[sf_value],
                    latency_norm_by_sf[sf_value],
                    metrics.energy_norm,
                    metrics.collision_norm,
                    reward_weights,
                    lambda_energy,
                    lambda_collision,
                )
                window_rewards.append(reward)
                raw_rows.append(
                    {
                        "network_size": network_size_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "cluster": node_clusters[node_id],
                        "round": round_id,
                        "node_id": node_id,
                        "sf": sf_value,
                        "window_start_s": window_start_s,
                        "node_offset_s": node_window["node_offset_s"],
                        "traffic_coeff": node_window["traffic_coeff"],
                        "rate_multiplier": node_window["rate_multiplier"],
                        "traffic_sent": traffic_sent,
                        "shadowing_db": node_window["shadowing_db"],
                        "shadowing_sigma_db": node_window["shadowing_sigma_db"],
                        "link_quality": node_window["link_quality"],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "collision_norm": metrics.collision_norm,
                        "reward": reward,
                    }
                )
                raw_rows.append(
                    {
                        "network_size": network_size_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "cluster": "all",
                        "round": round_id,
                        "node_id": node_id,
                        "sf": sf_value,
                        "window_start_s": window_start_s,
                        "node_offset_s": node_window["node_offset_s"],
                        "traffic_coeff": node_window["traffic_coeff"],
                        "rate_multiplier": node_window["rate_multiplier"],
                        "traffic_sent": traffic_sent,
                        "shadowing_db": node_window["shadowing_db"],
                        "shadowing_sigma_db": node_window["shadowing_sigma_db"],
                        "link_quality": node_window["link_quality"],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "collision_norm": metrics.collision_norm,
                        "reward": reward,
                    }
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
            logger.info(
                "Round %s - %s : récompense moyenne = %.4f",
                round_id,
                algo_label,
                avg_reward,
            )
            learning_curve_rows.append(
                {
                    "network_size": network_size_value,
                    "round": round_id,
                    "algo": algo_label,
                    "avg_reward": avg_reward,
                }
            )
    else:
        raise ValueError("algorithm doit être adr, mixra_h, mixra_opt ou ucb1_sf.")

    if output_dir is not None:
        write_simulation_results(output_dir, raw_rows, network_size=network_size_value)
        learning_curve_path = output_dir / "learning_curve.csv"
        learning_curve_header = ["network_size", "round", "algo", "avg_reward"]
        write_rows(
            learning_curve_path,
            learning_curve_header,
            [
                [row.get(key, "") for key in learning_curve_header]
                for row in learning_curve_rows
            ],
        )

    return Step2Result(
        raw_rows=raw_rows,
        selection_prob_rows=selection_prob_rows,
        learning_curve_rows=learning_curve_rows,
    )
