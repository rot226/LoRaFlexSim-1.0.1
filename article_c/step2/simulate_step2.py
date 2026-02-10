"""Simulation de l'étape 2 (proxy UCB1 et comparaisons)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import math
from statistics import mean, median, pstdev
from typing import Literal

from article_c.common.config import DEFAULT_CONFIG, STEP2_SAFE_CONFIG, Step2Config
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
    throughput_success: float
    energy_per_success: float


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


@dataclass
class RewardAlertState:
    consecutive_alerts: int = 0
    correction_rounds_left: int = 0
    collision_penalty_scale: float = 1.0
    reward_floor_boost: float = 0.0


logger = logging.getLogger(__name__)
_NO_CLAMP = False


def _clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _normalize_local(
    value: float, min_value: float, max_value: float, *, fallback: float = 0.5
) -> float:
    if math.isclose(min_value, max_value, abs_tol=1e-9):
        return _clip(fallback, 0.0, 1.0)
    return _clip((value - min_value) / (max_value - min_value), 0.0, 1.0)


def _compute_reward(
    success_rate: float,
    traffic_sent: int,
    sf_norm: float,
    latency_norm: float,
    energy_norm: float,
    collision_norm: float,
    throughput_success_norm: float,
    energy_per_success_norm: float,
    weights: AlgoRewardWeights,
    lambda_energy: float,
    lambda_collision: float,
    max_penalty_ratio: float,
    *,
    floor_on_zero_success: bool = False,
    log_components: bool = False,
    log_context: str | None = None,
) -> float:
    throughput_weight = 0.1
    energy_per_success_weight = 0.1
    reward_floor_base = 0.08
    latency_norm = _clip(latency_norm**1.35 * 1.08, 0.0, 1.0)
    energy_norm = _clip(energy_norm**1.35 * 1.08, 0.0, 1.0)
    collision_norm = _clip(collision_norm**1.15 * 1.04, 0.0, 1.0)
    energy_weight = weights.energy_weight * (1.0 + lambda_energy)
    total_weight = (
        weights.sf_weight
        + weights.latency_weight
        + energy_weight
        + throughput_weight
        + energy_per_success_weight
    )
    if total_weight <= 0:
        total_weight = 1.0
    sf_score = 1.0 - sf_norm
    latency_score = 1.0 - latency_norm
    energy_score = 1.0 - energy_norm
    throughput_score = _clip(throughput_success_norm, 0.0, 1.0)
    energy_per_success_score = 1.0 - _clip(energy_per_success_norm, 0.0, 1.0)
    weighted_quality = (
        weights.sf_weight * sf_score
        + weights.latency_weight * latency_score
        + energy_weight * energy_score
        + throughput_weight * throughput_score
        + energy_per_success_weight * energy_per_success_score
    ) / total_weight
    traffic_factor = 0.0
    if traffic_sent > 0:
        traffic_factor = _clip(traffic_sent / (traffic_sent + 20.0), 0.0, 1.0)
    collision_penalty = (
        (0.75 * lambda_collision)
        * weights.collision_weight
        * collision_norm
        * (0.7 + 0.3 * (1.0 - success_rate))
        * math.sqrt(max(success_rate, 0.0))
        * traffic_factor
    )
    if max_penalty_ratio >= 0.0:
        penalty_cap = max_penalty_ratio * weighted_quality
        if collision_penalty > penalty_cap:
            logger.info(
                "collision_penalty capped (collision_penalty=%.4f cap=%.4f).",
                collision_penalty,
                penalty_cap,
            )
            collision_penalty = penalty_cap
    if collision_penalty > success_rate * weighted_quality:
        logger.info(
            "Pénalité de collision dominante (collision_penalty=%.4f success_rate=%.4f weighted_quality=%.4f).",
            collision_penalty,
            success_rate,
            weighted_quality,
        )
    quality_term = weighted_quality * (0.6 + 0.4 * success_rate)
    success_term = 0.4 * success_rate
    bonus_quality_term = 0.05 * weighted_quality
    reward = quality_term + success_term + bonus_quality_term - collision_penalty
    reward_floor = max(weights.exploration_floor, 0.0)
    if floor_on_zero_success:
        reward_floor = max(reward_floor, reward_floor_base)
    if floor_on_zero_success and success_rate == 0.0 and reward_floor > 0.0:
        reward = reward_floor
    elif reward_floor > 0.0 and success_rate > 0.0 and reward < reward_floor:
        reward = reward_floor
    clipped_reward = _clip(reward, 0.0, 1.0)
    if reward_floor > 0.0 and clipped_reward < reward_floor:
        if success_rate > 0.0 or (floor_on_zero_success and success_rate <= 0.0):
            clipped_reward = reward_floor
    if clipped_reward == 0.0:
        logger.info(
            "Récompense nulle (success_rate=%.4f sf_norm=%.3f latency_norm=%.3f "
            "energy_norm=%.3f collision_norm=%.3f).",
            success_rate,
            sf_norm,
            latency_norm,
            energy_norm,
            collision_norm,
        )
    if log_components:
        suffix = f" ({log_context})" if log_context else ""
        logger.info(
            "reward components%s: success_rate=%.4f weighted_quality=%.4f "
            "collision_penalty=%.4f reward=%.4f sf_norm=%.3f latency_norm=%.3f "
            "energy_norm=%.3f collision_norm=%.3f throughput_norm=%.3f "
            "energy_per_success_norm=%.3f lambda_energy=%.3f lambda_collision=%.3f "
            "weights=(sf=%.2f latency=%.2f energy=%.2f collision=%.2f "
            "throughput=%.2f energy_per_success=%.2f).",
            suffix,
            success_rate,
            weighted_quality,
            collision_penalty,
            clipped_reward,
            sf_norm,
            latency_norm,
            energy_norm,
            collision_norm,
            throughput_success_norm,
            energy_per_success_norm,
            lambda_energy,
            lambda_collision,
            weights.sf_weight,
            weights.latency_weight,
            weights.energy_weight,
            weights.collision_weight,
            throughput_weight,
            energy_per_success_weight,
        )
    return clipped_reward


def _clamp_range(value: float, min_value: float, max_value: float) -> float:
    if _NO_CLAMP:
        return value
    return max(min_value, min(max_value, value))


def _apply_reward_floor_boost(
    weights: AlgoRewardWeights, reward_floor_boost: float
) -> AlgoRewardWeights:
    if reward_floor_boost <= 0.0:
        return weights
    boosted_floor = _clip(weights.exploration_floor + reward_floor_boost, 0.0, 1.0)
    return AlgoRewardWeights(
        sf_weight=weights.sf_weight,
        latency_weight=weights.latency_weight,
        energy_weight=weights.energy_weight,
        collision_weight=weights.collision_weight,
        exploration_floor=boosted_floor,
    )


def _consume_reward_alert_adjustment(
    alert_state: dict[tuple[int, str], RewardAlertState] | None,
    network_size: int,
    algo_label: str,
) -> tuple[float, float]:
    if alert_state is None:
        return 1.0, 0.0
    alert_key = (network_size, algo_label)
    state = alert_state.get(alert_key)
    if state is None or state.correction_rounds_left <= 0:
        return 1.0, 0.0
    state.correction_rounds_left -= 1
    collision_penalty_scale = state.collision_penalty_scale
    reward_floor_boost = state.reward_floor_boost
    if state.correction_rounds_left <= 0:
        state.collision_penalty_scale = 1.0
        state.reward_floor_boost = 0.0
    return collision_penalty_scale, reward_floor_boost


def _max_window_tx(
    window_duration_s: float,
    airtime_s: float,
    n_channels: int,
    safety_factor: float,
    *,
    min_tx: int = 1,
) -> int:
    if window_duration_s <= 0.0 or airtime_s <= 0.0:
        return min_tx
    effective_channels = max(n_channels, 1)
    denom = airtime_s * effective_channels * max(safety_factor, 0.1)
    if denom <= 0.0:
        return min_tx
    return max(min_tx, int(math.floor(window_duration_s / denom)))


def _apply_phase_offset(
    traffic_times: list[float],
    *,
    rng: random.Random,
    window_duration_s: float,
    base_period_s: float,
) -> list[float]:
    if not traffic_times or window_duration_s <= 0.0:
        return traffic_times
    max_offset = min(base_period_s, window_duration_s)
    if max_offset <= 0.0:
        return traffic_times
    phase_offset = rng.uniform(0.0, max_offset)
    return sorted(((time + phase_offset) % window_duration_s) for time in traffic_times)


def _assign_tx_channels(
    tx_starts: list[float],
    n_channels: int,
    *,
    rng: random.Random,
    mode: Literal["random", "round_robin"] = "random",
) -> list[int]:
    if not tx_starts:
        return []
    if n_channels <= 0:
        return [0 for _ in tx_starts]
    if mode == "round_robin":
        return [index % n_channels for index in range(len(tx_starts))]
    return [rng.randrange(n_channels) for _ in tx_starts]


def _effective_window_duration(
    tx_starts: list[float],
    airtime_s: float,
    fallback_duration_s: float,
) -> float:
    if not tx_starts:
        return fallback_duration_s
    min_start = min(tx_starts)
    max_end = max(tx_starts) + airtime_s
    duration = max_end - min_start
    if duration <= 0.0:
        return fallback_duration_s
    return duration


def _default_reference_size() -> int:
    sizes = list(DEFAULT_CONFIG.scenario.network_sizes)
    if not sizes:
        return 1
    return max(1, int(round(median(sizes))))


def _network_load_factor(
    network_size: int, reference_size: int, clamp_min: float, clamp_max: float
) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    return _clamp_range(ratio**0.4, clamp_min, clamp_max)


def _traffic_coeff_size_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    if ratio <= 1.0:
        return max(0.7, ratio**0.35)
    overload = ratio - 1.0
    softened_growth = 1.0 + 0.22 * math.log1p(overload)
    return _clamp_range(softened_growth, 1.0, 1.45)


def _traffic_coeff_variance_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    if ratio <= 1.0:
        return 1.0
    return _clamp_range(1.0 + 0.45 * (ratio - 1.0), 1.0, 2.2)


def _shadowing_sigma_size_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.2, network_size / reference_size)
    return _clamp_range(ratio**0.25, 0.85, 1.4)


def _link_quality_min_variation(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 0.02
    ratio = max(0.2, network_size / reference_size)
    if ratio <= 1.0:
        return 0.02
    growth = (ratio - 1.0) ** 0.75
    return _clamp_range(0.02 + 0.06 * growth, 0.02, 0.14)


def _link_quality_load_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.2, network_size / reference_size)
    if ratio <= 1.0:
        return _clamp_range(1.0 - 0.04 * (1.0 - ratio), 0.9, 1.0)
    overload = ratio - 1.0
    return _clamp_range(1.0 / (1.0 + 0.18 * overload**0.6), 0.65, 1.0)


def _link_quality_mean_degradation(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    overload = max(0.0, (network_size / reference_size) - 1.0)
    return _clamp_range(1.0 - 0.09 * math.log1p(overload), 0.7, 1.0)


def _link_quality_size_factor(network_size: int, reference_size: int) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.2, network_size / reference_size)
    if ratio <= 1.0:
        return _clamp_range(1.0 - 0.05 * (1.0 - ratio), 0.93, 1.0)
    overload = ratio - 1.0
    return _clamp_range(1.0 / (1.0 + 0.12 * overload), 0.7, 1.0)


def _collision_size_factor(
    network_size: int,
    reference_size: int,
    clamp_min: float,
    clamp_under_max: float,
    clamp_over_max: float,
) -> float:
    if reference_size <= 0:
        return 1.0
    ratio = max(0.1, network_size / reference_size)
    if ratio <= 1.0:
        return _clamp_range(ratio**0.4, clamp_min, clamp_under_max)
    overload = ratio - 1.0
    return _clamp_range(
        1.0 + 0.6 * (1.0 + overload) ** 0.45, 1.0, clamp_over_max
    )


def _resolve_load_clamps(
    step2_defaults: Step2Config,
    network_load_min: float | None,
    network_load_max: float | None,
    *,
    safe_profile: bool,
    no_clamp: bool = False,
) -> tuple[float, float]:
    load_clamp_min_value = (
        step2_defaults.network_load_min
        if network_load_min is None
        else float(network_load_min)
    )
    load_clamp_max_value = (
        step2_defaults.network_load_max
        if network_load_max is None
        else float(network_load_max)
    )
    if load_clamp_min_value > load_clamp_max_value:
        load_clamp_min_value, load_clamp_max_value = (
            load_clamp_max_value,
            load_clamp_min_value,
        )
    if no_clamp:
        return load_clamp_min_value, load_clamp_max_value
    if safe_profile:
        load_clamp_min_value = max(
            load_clamp_min_value, STEP2_SAFE_CONFIG.network_load_min
        )
        load_clamp_max_value = min(
            load_clamp_max_value, STEP2_SAFE_CONFIG.network_load_max
        )
        if load_clamp_min_value > load_clamp_max_value:
            load_clamp_min_value = STEP2_SAFE_CONFIG.network_load_min
            load_clamp_max_value = STEP2_SAFE_CONFIG.network_load_max
    return load_clamp_min_value, load_clamp_max_value


def _resolve_collision_clamps(
    step2_defaults: Step2Config,
    collision_size_min: float | None,
    collision_size_under_max: float | None,
    collision_size_over_max: float | None,
    *,
    safe_profile: bool,
    no_clamp: bool = False,
) -> tuple[float, float, float]:
    collision_clamp_min_value = (
        step2_defaults.collision_size_min
        if collision_size_min is None
        else float(collision_size_min)
    )
    collision_clamp_under_max_value = (
        step2_defaults.collision_size_under_max
        if collision_size_under_max is None
        else float(collision_size_under_max)
    )
    collision_clamp_over_max_value = (
        step2_defaults.collision_size_over_max
        if collision_size_over_max is None
        else float(collision_size_over_max)
    )
    if collision_clamp_min_value > collision_clamp_under_max_value:
        collision_clamp_min_value, collision_clamp_under_max_value = (
            collision_clamp_under_max_value,
            collision_clamp_min_value,
        )
    if collision_clamp_under_max_value > collision_clamp_over_max_value:
        collision_clamp_under_max_value, collision_clamp_over_max_value = (
            collision_clamp_over_max_value,
            collision_clamp_under_max_value,
        )
    if no_clamp:
        return (
            collision_clamp_min_value,
            collision_clamp_under_max_value,
            collision_clamp_over_max_value,
        )
    if safe_profile:
        collision_clamp_min_value = max(
            collision_clamp_min_value, STEP2_SAFE_CONFIG.collision_size_min
        )
        collision_clamp_under_max_value = min(
            collision_clamp_under_max_value,
            STEP2_SAFE_CONFIG.collision_size_under_max,
        )
        collision_clamp_over_max_value = min(
            collision_clamp_over_max_value, STEP2_SAFE_CONFIG.collision_size_over_max
        )
        if collision_clamp_min_value > collision_clamp_under_max_value:
            collision_clamp_min_value = STEP2_SAFE_CONFIG.collision_size_min
            collision_clamp_under_max_value = STEP2_SAFE_CONFIG.collision_size_under_max
        if collision_clamp_under_max_value > collision_clamp_over_max_value:
            collision_clamp_over_max_value = STEP2_SAFE_CONFIG.collision_size_over_max
            if collision_clamp_under_max_value > collision_clamp_over_max_value:
                collision_clamp_under_max_value = (
                    STEP2_SAFE_CONFIG.collision_size_under_max
                )
    return (
        collision_clamp_min_value,
        collision_clamp_under_max_value,
        collision_clamp_over_max_value,
    )


def _log_size_factor_comparison(
    network_size: int,
    reference_size: int,
    *,
    load_factor: float,
    collision_size_factor: float,
    legacy_load_factor: float,
    legacy_collision_size_factor: float,
    load_clamp_min: float,
    load_clamp_max: float,
    collision_clamp_min: float,
    collision_clamp_under_max: float,
    collision_clamp_over_max: float,
) -> None:
    logger.info(
        "Taille %s (réf=%s) facteurs: charge=%.3f (avant=%.3f, clamp %.2f..%.2f) "
        "collision=%.3f (avant=%.3f, clamp %.2f..%.2f/%.2f).",
        network_size,
        reference_size,
        load_factor,
        legacy_load_factor,
        load_clamp_min,
        load_clamp_max,
        collision_size_factor,
        legacy_collision_size_factor,
        collision_clamp_min,
        collision_clamp_under_max,
        collision_clamp_over_max,
    )


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


def _congestion_collision_probability(
    network_size: int,
    reference_size: int,
    *,
    base_coeff: float,
    growth_coeff: float,
    max_probability: float,
) -> float:
    if reference_size <= 0:
        return 0.0
    overload = max(0.0, (network_size / reference_size) - 1.0)
    return _clip(
        base_coeff * (1.0 - math.exp(-growth_coeff * overload)),
        0.0,
        max_probability,
    )


def _compute_window_metrics(
    successes: int,
    traffic_sent: int,
    bitrate_norm: float,
    energy_norm: float,
    collision_norm: float,
    *,
    payload_bytes: int,
    effective_duration_s: float,
    window_duration_s: float,
    airtime_s: float,
) -> WindowMetrics:
    success_rate = successes / traffic_sent if traffic_sent > 0 else 0.0
    throughput_duration_s = window_duration_s if window_duration_s > 0 else effective_duration_s
    throughput_success = (
        successes * payload_bytes / throughput_duration_s if throughput_duration_s > 0 else 0.0
    )
    energy_per_success = airtime_s * traffic_sent / max(successes, 1)
    return WindowMetrics(
        success_rate=success_rate,
        bitrate_norm=bitrate_norm,
        energy_norm=energy_norm,
        collision_norm=collision_norm,
        throughput_success=throughput_success,
        energy_per_success=energy_per_success,
    )


def _compute_collision_successes(
    transmissions: dict[int, dict[int, list[tuple[float, float, int]]]],
    *,
    rng: random.Random | None = None,
    approx_threshold: int = 5000,
    approx_sample_size: int = 2500,
    capture_probability: float = DEFAULT_CONFIG.step2.capture_probability,
) -> tuple[dict[int, int], dict[int, int], int, bool]:
    def _compute_collisions(
        events: list[tuple[float, float, int]],
        *,
        rng: random.Random,
        capture_probability: float,
    ) -> list[bool]:
        collided = [False] * len(events)
        indexed_events = sorted(enumerate(events), key=lambda item: item[1][0])
        group_indices: list[int] = []
        group_end: float | None = None
        for event_index, (start, end, _node_id) in indexed_events:
            if group_end is None or start >= group_end:
                if len(group_indices) > 1:
                    if rng.random() < capture_probability:
                        survivor = rng.choice(group_indices)
                        for index in group_indices:
                            collided[index] = index != survivor
                    else:
                        for index in group_indices:
                            collided[index] = True
                group_indices = [event_index]
                group_end = end
            else:
                group_indices.append(event_index)
                if end > group_end:
                    group_end = end
        if len(group_indices) > 1:
            if rng.random() < capture_probability:
                survivor = rng.choice(group_indices)
                for index in group_indices:
                    collided[index] = index != survivor
            else:
                for index in group_indices:
                    collided[index] = True
        return collided

    transmissions_per_bucket = {
        (channel_id, sf_value): len(events)
        for channel_id, sf_events in transmissions.items()
        for sf_value, events in sf_events.items()
    }
    per_node_total_global: dict[int, int] = {}
    for sf_events in transmissions.values():
        for events in sf_events.values():
            for _start, _end, node_id in events:
                per_node_total_global[node_id] = per_node_total_global.get(node_id, 0) + 1
    max_transmissions_per_bucket = max(transmissions_per_bucket.values(), default=0)
    approx_mode = max_transmissions_per_bucket > approx_threshold
    rng = rng or random.Random(0)
    successes_by_node: dict[int, int] = {}
    for channel_id, sf_events in transmissions.items():
        for sf_value, events in sf_events.items():
            if not events:
                continue
            bucket_total = transmissions_per_bucket.get((channel_id, sf_value), 0)
            bucket_approx = bucket_total > approx_threshold
            per_node_total: dict[int, int] = {}
            for _start, _end, node_id in events:
                per_node_total[node_id] = per_node_total.get(node_id, 0) + 1
            sample_probability = 1.0
            sampled_events = events
            if bucket_approx and bucket_total > 0:
                scaled_sample_size = min(max(1, approx_sample_size), len(events))
                if len(events) > scaled_sample_size:
                    sample_probability = scaled_sample_size / len(events)
                    sampled_events = [
                        event for event in events if rng.random() < sample_probability
                    ]
                    if not sampled_events:
                        sampled_events = [rng.choice(events)]
            collided = _compute_collisions(
                sampled_events,
                rng=rng,
                capture_probability=capture_probability,
            )
            sample_successes: dict[int, int] = {}
            for event_index, (_start, _end, node_id) in enumerate(sampled_events):
                if not collided[event_index]:
                    sample_successes[node_id] = sample_successes.get(node_id, 0) + 1
            if sample_probability < 1.0:
                for node_id, successes in sample_successes.items():
                    estimated = int(round(successes / sample_probability))
                    remaining_global = per_node_total_global.get(
                        node_id, 0
                    ) - successes_by_node.get(node_id, 0)
                    if remaining_global <= 0:
                        continue
                    estimate_cap = min(per_node_total[node_id], remaining_global)
                    successes_by_node[node_id] = successes_by_node.get(node_id, 0) + min(
                        estimated, estimate_cap
                    )
            else:
                for node_id, successes in sample_successes.items():
                    remaining_global = per_node_total_global.get(
                        node_id, 0
                    ) - successes_by_node.get(node_id, 0)
                    if remaining_global <= 0:
                        continue
                    successes_by_node[node_id] = (
                        successes_by_node.get(node_id, 0)
                        + min(successes, per_node_total[node_id], remaining_global)
                    )
    return successes_by_node, per_node_total_global, max_transmissions_per_bucket, approx_mode


def _collect_traffic_sent(node_windows: list[dict[str, object]]) -> dict[int, int]:
    traffic_sent_by_node: dict[int, int] = {}
    for node_window in node_windows:
        node_id = int(node_window["node_id"])
        traffic_sent = int(node_window["traffic_sent"])
        traffic_sent_by_node[node_id] = traffic_sent_by_node.get(node_id, 0) + traffic_sent
    return traffic_sent_by_node


def _compute_successes_and_traffic(
    node_windows: list[dict[str, object]],
    airtime_by_sf: dict[int, float],
    *,
    rng: random.Random,
    capture_probability: float,
    approx_threshold: int = 5000,
    approx_sample_size: int = 2500,
    debug_step2: bool = False,
) -> tuple[dict[int, int], dict[int, int], int, bool]:
    transmissions_by_channel: dict[int, dict[int, list[tuple[float, float, int]]]] = {}
    for node_window in node_windows:
        sf_value = int(node_window["sf"])
        airtime = airtime_by_sf[sf_value]
        tx_starts = node_window["tx_starts"]
        node_id = int(node_window["node_id"])
        tx_channels = node_window["tx_channels"]
        for start_time, channel_id in zip(tx_starts, tx_channels):
            transmissions_by_channel.setdefault(channel_id, {}).setdefault(sf_value, []).append(
                (start_time, start_time + airtime, node_id)
            )
    successes_by_node, per_node_total_global, transmission_count, approx_mode = (
        _compute_collision_successes(
            transmissions_by_channel,
            rng=rng,
            approx_threshold=approx_threshold,
            approx_sample_size=approx_sample_size,
            capture_probability=capture_probability,
        )
    )
    traffic_sent_by_node = _collect_traffic_sent(node_windows)
    aberrant_nodes: set[int] = set()
    for node_id in (
        set(per_node_total_global) | set(traffic_sent_by_node) | set(successes_by_node)
    ):
        traffic_sent = traffic_sent_by_node.get(node_id, 0)
        per_node_total = per_node_total_global.get(node_id, 0)
        if traffic_sent > per_node_total:
            traffic_sent_by_node[node_id] = per_node_total
            traffic_sent = per_node_total
        if debug_step2 and traffic_sent != per_node_total:
            aberrant_nodes.add(node_id)
    for node_id, successes in successes_by_node.items():
        traffic_sent = traffic_sent_by_node.get(node_id, 0)
        per_node_total = per_node_total_global.get(node_id, 0)
        corrected_successes = min(successes, traffic_sent, per_node_total)
        if corrected_successes != successes:
            logger.warning(
                (
                    "Succès incohérents après collisions "
                    "(node_id=%s successes=%s traffic_sent=%s per_node_total=%s)."
                ),
                node_id,
                successes,
                traffic_sent,
                per_node_total,
            )
            aberrant_nodes.add(node_id)
            successes_by_node[node_id] = corrected_successes
        elif debug_step2 and traffic_sent != per_node_total:
            aberrant_nodes.add(node_id)
    if debug_step2 and aberrant_nodes:
        for node_id in sorted(aberrant_nodes):
            logger.debug(
                "Noeud aberrant (node_id=%s per_node_total=%s traffic_sent=%s successes=%s).",
                node_id,
                per_node_total_global.get(node_id, 0),
                traffic_sent_by_node.get(node_id, 0),
                successes_by_node.get(node_id, 0),
            )
    return successes_by_node, traffic_sent_by_node, transmission_count, approx_mode


def _summarize_values(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), median(values), max(values)


def _sample_node_windows(
    node_windows: list[dict[str, object]],
    *,
    rng: random.Random,
    max_nodes: int = 12,
    min_nodes: int = 2,
    ratio: float = 0.2,
) -> list[dict[str, object]]:
    if not node_windows:
        return []
    if len(node_windows) <= min_nodes:
        return list(node_windows)
    target = max(min_nodes, int(round(len(node_windows) * ratio)))
    target = min(max_nodes, max(min_nodes, target))
    if target >= len(node_windows):
        return list(node_windows)
    return rng.sample(node_windows, k=target)


def _approx_collision_gap_ratio(
    node_windows: list[dict[str, object]],
    airtime_by_sf: dict[int, float],
    *,
    capture_probability: float,
    approx_threshold: int,
    approx_sample_size: int,
    seed: int,
    round_id: int,
    max_rounds: int,
) -> dict[str, float] | None:
    if round_id >= max_rounds or not node_windows:
        return None
    subset_rng = random.Random(seed + 101 + round_id)
    subset_windows = _sample_node_windows(node_windows, rng=subset_rng)
    if not subset_windows:
        return None
    approx_rng = random.Random(seed + 313 + round_id)
    exact_rng = random.Random(seed + 313 + round_id)
    approx_successes, approx_traffic, _, _ = _compute_successes_and_traffic(
        subset_windows,
        airtime_by_sf,
        rng=approx_rng,
        capture_probability=capture_probability,
        approx_threshold=approx_threshold,
        approx_sample_size=approx_sample_size,
    )
    exact_successes, exact_traffic, _, _ = _compute_successes_and_traffic(
        subset_windows,
        airtime_by_sf,
        rng=exact_rng,
        capture_probability=capture_probability,
        approx_threshold=10**9,
        approx_sample_size=approx_sample_size,
    )
    total_traffic = sum(exact_traffic.values())
    if total_traffic <= 0:
        return None
    approx_ratio = sum(approx_successes.values()) / total_traffic
    exact_ratio = sum(exact_successes.values()) / total_traffic
    gap = abs(approx_ratio - exact_ratio)
    gap_ratio = gap / max(exact_ratio, 1e-6)
    return {
        "subset_nodes": float(len(subset_windows)),
        "approx_ratio": approx_ratio,
        "exact_ratio": exact_ratio,
        "gap_ratio": gap_ratio,
    }


def _should_debug_log(debug_step2: bool, round_id: int, max_rounds: int = 3) -> bool:
    return debug_step2 and round_id < max_rounds


def _log_debug_stage(
    *,
    stage: str,
    network_size: int,
    algo_label: str,
    round_id: int,
    details: str,
) -> None:
    logger.info(
        "Debug step2 [%s] - taille=%s algo=%s round=%s %s",
        stage,
        network_size,
        algo_label,
        round_id,
        details,
    )


def _log_reward_stats(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    rewards: list[float],
    reward_alert_level: int,
    log_counts: dict[tuple[int, str, int], int] | None = None,
    alert_state: dict[tuple[int, str], "RewardAlertState"] | None = None,
    global_alert_counts: dict[tuple[int, str], int] | None = None,
    reward_floor: float = 0.0,
) -> None:
    log_key = (network_size, algo_label, round_id)
    if log_counts is not None:
        log_counts[log_key] = log_counts.get(log_key, 0) + 1
        if log_counts[log_key] > 1:
            return
    min_reward, median_reward, max_reward = _summarize_values(rewards)
    std_reward = pstdev(rewards)
    alert_key = (network_size, algo_label)
    uniform_reward_alert = math.isclose(min_reward, max_reward, abs_tol=1e-6)
    if math.isclose(std_reward, 0.0, abs_tol=1e-6):
        logger.warning(
            "Variance reward nulle (taille=%s algo=%s round=%s).",
            network_size,
            algo_label,
            round_id,
        )
    if uniform_reward_alert:
        if global_alert_counts is not None:
            global_alert_counts[alert_key] = global_alert_counts.get(alert_key, 0) + 1
        should_log_alert = True
        if global_alert_counts is not None:
            should_log_alert = global_alert_counts[alert_key] <= 3
        if should_log_alert:
            logger.log(
                reward_alert_level,
                "Alerte reward uniforme (taille=%s algo=%s round=%s reward=%.4f).",
                network_size,
                algo_label,
                round_id,
                min_reward,
            )
    if alert_state is not None:
        state = alert_state.setdefault(alert_key, RewardAlertState())
        if uniform_reward_alert:
            state.consecutive_alerts += 1
        else:
            state.consecutive_alerts = 0
        if state.consecutive_alerts >= 3 and state.correction_rounds_left == 0:
            if reward_floor < 0.05:
                state.reward_floor_boost = 0.05
                state.collision_penalty_scale = 1.0
                correction = "hausse du reward_floor"
            else:
                state.collision_penalty_scale = 0.75
                state.reward_floor_boost = 0.0
                correction = "réduction du collision_penalty"
            state.correction_rounds_left = 3
            logger.info(
                "Correction reward appliquée (taille=%s algo=%s round=%s %s, durée=%s rounds).",
                network_size,
                algo_label,
                round_id,
                correction,
                state.correction_rounds_left,
            )
    logger.info(
        "Stats reward [#%s] - taille=%s algo=%s round=%s reward[min/med/max]=%.4f/%.4f/%.4f",
        log_counts.get(log_key, 1) if log_counts is not None else 1,
        network_size,
        algo_label,
        round_id,
        min_reward,
        median_reward,
        max_reward,
    )


def _log_collision_stability(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    collision_norms: list[float],
    approx_collision_mode: bool,
    sensitivity_threshold: float = 0.6,
) -> None:
    if not collision_norms:
        return
    min_collision, median_collision, max_collision = _summarize_values(collision_norms)
    spread_ratio = 0.0
    if max_collision > 0.0:
        spread_ratio = (max_collision - min_collision) / max_collision
    logger.debug(
        "Sensibilité collisions - taille=%s algo=%s round=%s "
        "ratio=%.3f min/med/max=%.3f/%.3f/%.3f approx=%s",
        network_size,
        algo_label,
        round_id,
        spread_ratio,
        min_collision,
        median_collision,
        max_collision,
        approx_collision_mode,
    )
    if spread_ratio >= sensitivity_threshold:
        logger.info(
            "Stabilité collisions à surveiller (taille=%s algo=%s round=%s ratio=%.3f).",
            network_size,
            algo_label,
            round_id,
            spread_ratio,
        )


def _log_round_traffic_debug(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    traffic_sent_total: int,
    successes_total: int,
) -> None:
    logger.info(
        "Debug step2 trafic - taille=%s algo=%s round=%s traffic_sent=%s successes=%s",
        network_size,
        algo_label,
        round_id,
        traffic_sent_total,
        successes_total,
    )


def _log_congestion_probability(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    congestion_probability: float,
    congestion_coeff: float,
    congestion_coeff_base: float,
    congestion_coeff_growth: float,
    congestion_coeff_max: float,
) -> None:
    logger.info(
        "Congestion (taille=%s algo=%s round=%s) p=%.3f "
        "(coeff=%.2f base=%.3f growth=%.3f max=%.3f).",
        network_size,
        algo_label,
        round_id,
        congestion_probability,
        congestion_coeff,
        congestion_coeff_base,
        congestion_coeff_growth,
        congestion_coeff_max,
    )


def _log_cluster_all_diagnostics(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    traffic_sent: list[int],
    successes: list[int],
    collision_norms: list[float],
    link_qualities: list[float],
    rewards: list[float],
    lambda_collision: float,
    congestion_probability: float,
    collision_size_factor: float,
) -> None:
    traffic_total = sum(traffic_sent)
    traffic_per_node = traffic_total / len(traffic_sent) if traffic_sent else 0.0
    successes_total = sum(successes)
    min_sent, median_sent, max_sent = _summarize_values(
        [float(value) for value in traffic_sent]
    )
    min_success, median_success, max_success = _summarize_values(
        [float(value) for value in successes]
    )
    min_collision, median_collision, max_collision = _summarize_values(collision_norms)
    min_lq, median_lq, max_lq = _summarize_values(link_qualities)
    min_reward, median_reward, max_reward = _summarize_values(rewards)
    logger.info(
        "Diag cluster=all - taille=%s algo=%s round=%s "
        "traffic_sent[total/min/med/max]=%s/%.0f/%.1f/%.0f "
        "traffic_sent_per_node=%.2f "
        "successes[total/min/med/max]=%s/%.0f/%.1f/%.0f "
        "collision_norm[min/med/max]=%.3f/%.3f/%.3f "
        "link_quality[min/med/max]=%.3f/%.3f/%.3f "
        "reward[min/med/max]=%.4f/%.4f/%.4f "
        "(lambda_collision=%.3f congestion_probability=%.3f collision_size_factor=%.3f).",
        network_size,
        algo_label,
        round_id,
        traffic_total,
        min_sent,
        median_sent,
        max_sent,
        traffic_per_node,
        successes_total,
        min_success,
        median_success,
        max_success,
        min_collision,
        median_collision,
        max_collision,
        min_lq,
        median_lq,
        max_lq,
        min_reward,
        median_reward,
        max_reward,
        lambda_collision,
        congestion_probability,
        collision_size_factor,
    )


def _log_pre_collision_stats(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    traffic_sent_by_node: dict[int, int],
    link_qualities: list[float],
) -> None:
    min_sent, median_sent, max_sent = _summarize_values(
        [float(value) for value in traffic_sent_by_node.values()]
    )
    min_lq, median_lq, max_lq = _summarize_values(link_qualities)
    logger.info(
        "Pré-collision - taille=%s algo=%s round=%s traffic_sent[min/med/max]=%.0f/%.1f/%.0f "
        "link_quality[min/med/max]=%.3f/%.3f/%.3f",
        network_size,
        algo_label,
        round_id,
        min_sent,
        median_sent,
        max_sent,
        min_lq,
        median_lq,
        max_lq,
    )


def _log_pre_clip_stats(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    successes_by_node: dict[int, int],
    traffic_sent_by_node: dict[int, int],
) -> None:
    min_success, median_success, max_success = _summarize_values(
        [float(value) for value in successes_by_node.values()]
    )
    min_sent, median_sent, max_sent = _summarize_values(
        [float(value) for value in traffic_sent_by_node.values()]
    )
    logger.info(
        "Pré-clipping - taille=%s algo=%s round=%s successes[min/med/max]=%.0f/%.1f/%.0f "
        "traffic_sent[min/med/max]=%.0f/%.1f/%.0f",
        network_size,
        algo_label,
        round_id,
        min_success,
        median_success,
        max_success,
        min_sent,
        median_sent,
        max_sent,
    )


def _log_loss_breakdown(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    losses_collisions: int,
    losses_congestion: int,
    losses_link_quality: int,
) -> None:
    dominant_cause = _dominant_loss_cause(
        losses_collisions=losses_collisions,
        losses_congestion=losses_congestion,
        losses_link_quality=losses_link_quality,
    )
    logger.info(
        "Pertes - taille=%s algo=%s round=%s collisions=%s congestion=%s link_quality=%s "
        "dominante=%s",
        network_size,
        algo_label,
        round_id,
        losses_collisions,
        losses_congestion,
        losses_link_quality,
        dominant_cause,
    )


def _dominant_loss_cause(
    *,
    losses_collisions: int,
    losses_congestion: int,
    losses_link_quality: int,
) -> str:
    losses = {
        "collisions": losses_collisions,
        "congestion": losses_congestion,
        "link_quality": losses_link_quality,
    }
    total_losses = sum(losses.values())
    if total_losses <= 0:
        return "aucune"
    return max(losses, key=losses.get)


def _log_success_ratio_summary(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    total_traffic_sent: int,
    ratio_after_collisions: float,
    ratio_after_congestion: float,
    ratio_after_link: float,
    losses_collisions: int,
    losses_congestion: int,
    losses_link_quality: int,
) -> None:
    dominant_cause = _dominant_loss_cause(
        losses_collisions=losses_collisions,
        losses_congestion=losses_congestion,
        losses_link_quality=losses_link_quality,
    )
    logger.info(
        "Résumé round - taille=%s algo=%s round=%s traffic_sent=%s "
        "ratios_succes[collisions/congestion/lien]=%.3f/%.3f/%.3f "
        "pertes[collisions/congestion/lien]=%s/%s/%s dominante=%s",
        network_size,
        algo_label,
        round_id,
        total_traffic_sent,
        ratio_after_collisions,
        ratio_after_congestion,
        ratio_after_link,
        losses_collisions,
        losses_congestion,
        losses_link_quality,
        dominant_cause,
    )


def _log_success_chain(
    *,
    network_size: int,
    n_nodes: int,
    algo_label: str,
    round_id: int,
    total_traffic_sent: int,
    successes_after_collisions: int,
    successes_after_congestion: int,
    successes_after_link: int,
) -> None:
    logger.info(
        "Chaîne succès - taille=%s algo=%s round=%s traffic_sent=%s "
        "apres_collisions=%s apres_congestion=%s apres_link_quality=%s",
        network_size,
        algo_label,
        round_id,
        total_traffic_sent,
        successes_after_collisions,
        successes_after_congestion,
        successes_after_link,
    )
    avg_traffic_per_node = total_traffic_sent / max(n_nodes, 1)
    logger.info(
        "Charge moyenne - taille=%s algo=%s round=%s traffic_sent_total/n_nodes=%.2f",
        network_size,
        algo_label,
        round_id,
        avg_traffic_per_node,
    )


def _log_clamp_ratio_and_adjust(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    clamp_ratios: list[float],
    clamp_flags: list[bool],
    traffic_coeff_scale: float,
    window_duration_s: float,
    tx_window_safety_factor: float,
) -> tuple[float, float, float]:
    if not clamp_ratios:
        return traffic_coeff_scale, window_duration_s, tx_window_safety_factor
    clamped_count = sum(1 for flagged in clamp_flags if flagged)
    total_nodes = len(clamp_ratios)
    clamp_share = clamped_count / max(total_nodes, 1)
    min_ratio, median_ratio, max_ratio = _summarize_values(clamp_ratios)
    logger.info(
        "Clamp ratio round=%s taille=%s algo=%s ratio[min/med/max]=%.3f/%.3f/%.3f "
        "noeuds_clampes=%s/%s (%.1f%%).",
        round_id,
        network_size,
        algo_label,
        min_ratio,
        median_ratio,
        max_ratio,
        clamped_count,
        total_nodes,
        clamp_share * 100.0,
    )
    if clamp_share < 0.5:
        return traffic_coeff_scale, window_duration_s, tx_window_safety_factor
    logger.warning(
        "Majorité de noeuds clampée (round=%s taille=%s algo=%s).",
        round_id,
        network_size,
        algo_label,
    )
    adjust_factor = max(0.6, min(0.95, median_ratio))
    new_traffic_coeff_scale = max(0.1, traffic_coeff_scale * adjust_factor)
    new_tx_window_safety_factor = max(0.7, tx_window_safety_factor * adjust_factor)
    window_boost = min(1.3, max(1.0, 1.0 / max(median_ratio, 0.5)))
    new_window_duration_s = window_duration_s * window_boost
    if (
        math.isclose(new_traffic_coeff_scale, traffic_coeff_scale, rel_tol=1e-6)
        and math.isclose(new_tx_window_safety_factor, tx_window_safety_factor, rel_tol=1e-6)
        and math.isclose(new_window_duration_s, window_duration_s, rel_tol=1e-6)
    ):
        return traffic_coeff_scale, window_duration_s, tx_window_safety_factor
    logger.info(
        "Ajustement clamp round=%s taille=%s algo=%s traffic_coeff_scale=%.3f→%.3f "
        "window_duration_s=%.2f→%.2f tx_window_safety_factor=%.2f→%.2f.",
        round_id,
        network_size,
        algo_label,
        traffic_coeff_scale,
        new_traffic_coeff_scale,
        window_duration_s,
        new_window_duration_s,
        tx_window_safety_factor,
        new_tx_window_safety_factor,
    )
    return (
        new_traffic_coeff_scale,
        new_window_duration_s,
        new_tx_window_safety_factor,
    )


def _apply_collision_control(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    collision_norms: list[float],
    traffic_coeff_scale: float,
    window_duration_s: float,
    threshold: float = 0.65,
    traffic_scale_floor: float = 0.2,
    traffic_scale_ceiling: float = 2.5,
    window_boost_max: float = 1.5,
) -> tuple[float, float]:
    if not collision_norms:
        return traffic_coeff_scale, window_duration_s
    avg_collision = mean(collision_norms)
    if avg_collision <= threshold:
        return traffic_coeff_scale, window_duration_s
    severity = _clip((avg_collision - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    reduction = 1.0 - 0.2 * severity
    increase = 1.0 + 0.25 * severity
    new_traffic_coeff_scale = _clamp_range(
        traffic_coeff_scale * reduction, traffic_scale_floor, traffic_scale_ceiling
    )
    new_window_duration_s = min(window_duration_s * increase, window_duration_s * window_boost_max)
    if (
        math.isclose(new_traffic_coeff_scale, traffic_coeff_scale, rel_tol=1e-6)
        and math.isclose(new_window_duration_s, window_duration_s, rel_tol=1e-6)
    ):
        return traffic_coeff_scale, window_duration_s
    logger.warning(
        "Contrôle collisions (round=%s taille=%s algo=%s collision_norm_moy=%.3f "
        "seuil=%.2f): traffic_coeff_scale=%.3f→%.3f window_duration_s=%.2f→%.2f.",
        round_id,
        network_size,
        algo_label,
        avg_collision,
        threshold,
        traffic_coeff_scale,
        new_traffic_coeff_scale,
        window_duration_s,
        new_window_duration_s,
    )
    return new_traffic_coeff_scale, new_window_duration_s


def _sample_log_normal_shadowing(
    rng: random.Random, mean_db: float, sigma_db: float
) -> tuple[float, float]:
    shadowing_db = rng.gauss(mean_db, sigma_db)
    return shadowing_db, 10 ** (-shadowing_db / 10.0)


def _log_link_quality_summary(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    link_qualities: list[float],
) -> None:
    min_lq, median_lq, max_lq = _summarize_values(link_qualities)
    logger.debug(
        "Résumé link_quality - taille=%s algo=%s round=%s link_quality[min/med/max]=%.3f/%.3f/%.3f",
        network_size,
        algo_label,
        round_id,
        min_lq,
        median_lq,
        max_lq,
    )


def _log_link_quality_stats(
    *,
    network_size: int,
    algo_label: str,
    round_id: int,
    link_qualities: list[float],
) -> None:
    if not link_qualities:
        return
    mean_lq = mean(link_qualities)
    std_lq = pstdev(link_qualities)
    logger.info(
        "Stats link_quality - taille=%s algo=%s round=%s moyenne=%.3f ecart_type=%.3f",
        network_size,
        algo_label,
        round_id,
        mean_lq,
        std_lq,
    )


def _log_link_quality_size_summary(
    *,
    network_size: int,
    algo_label: str,
    link_qualities: list[float],
) -> None:
    if not link_qualities:
        return
    mean_lq = mean(link_qualities)
    std_lq = pstdev(link_qualities)
    logger.info(
        "Synthèse link_quality - taille=%s algo=%s moyenne=%.3f ecart_type=%.3f n=%s",
        network_size,
        algo_label,
        mean_lq,
        std_lq,
        len(link_qualities),
    )


def _log_value_distribution(
    *,
    label: str,
    network_size: int,
    algo_label: str,
    values: list[float],
    round_id: int | None = None,
) -> None:
    if not values:
        return
    min_value, median_value, max_value = _summarize_values(values)
    mean_value = mean(values)
    std_value = pstdev(values)
    round_suffix = f" round={round_id}" if round_id is not None else ""
    logger.info(
        "Distribution %s - taille=%s algo=%s%s min/med/max=%.3f/%.3f/%.3f moyenne=%.3f ecart_type=%.3f n=%s",
        label,
        network_size,
        algo_label,
        round_suffix,
        min_value,
        median_value,
        max_value,
        mean_value,
        std_value,
        len(values),
    )


def _apply_link_quality_variation(
    rng: random.Random,
    link_quality: float,
    *,
    network_size: int,
    reference_size: int,
) -> float:
    load_factor = _link_quality_load_factor(network_size, reference_size)
    mean_degradation = _link_quality_mean_degradation(network_size, reference_size)
    size_factor = _link_quality_size_factor(network_size, reference_size)
    link_quality = _clip(
        link_quality * load_factor * mean_degradation * size_factor, 0.0, 1.0
    )
    variation = rng.gauss(1.0, 0.12)
    min_variation = _link_quality_min_variation(network_size, reference_size)
    variation_floor = 0.2
    if variation < variation_floor:
        variation = variation_floor
    deviation = variation - 1.0
    if abs(deviation) < min_variation:
        if deviation == 0.0:
            deviation = min_variation if rng.random() < 0.5 else -min_variation
        else:
            deviation = math.copysign(min_variation, deviation)
        variation = 1.0 + deviation
    return _clip(link_quality * variation, 0.0, 1.0)


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


def _reward_weights_for_algo(
    algorithm: str, reward_floor: float | None = None
) -> AlgoRewardWeights:
    default_floor = 0.02
    # Poids harmonisés pour limiter les incohérences entre algorithmes :
    # - sf_weight : privilégie un SF faible (meilleur débit, moins d'airtime).
    # - latency_weight : pénalise la latence (fenêtres plus lentes, congestion).
    # - energy_weight : pénalise la consommation énergétique (airtime/puissance).
    # - collision_weight : renforce la pénalité sur les collisions.
    weights = AlgoRewardWeights(
        sf_weight=0.36,
        latency_weight=0.3,
        energy_weight=0.34,
        collision_weight=0.14,
    )
    if algorithm not in ("adr", "mixra_h", "mixra_opt"):
        default_floor = 0.1
    selected_floor = default_floor if reward_floor is None else reward_floor
    if selected_floor > 0.0:
        return AlgoRewardWeights(
            sf_weight=weights.sf_weight,
            latency_weight=weights.latency_weight,
            energy_weight=weights.energy_weight,
            collision_weight=weights.collision_weight,
            exploration_floor=selected_floor,
        )
    return weights


def _compute_collision_norm(
    *,
    airtime_norm: float,
    congestion_probability: float,
    collision_size_factor: float,
    successes: int,
    traffic_sent: int,
) -> float:
    successes = min(successes, traffic_sent)
    success_ratio = successes / traffic_sent if traffic_sent > 0 else 0.0
    congestion_scale = 1.0 + 0.65 * congestion_probability
    size_scale = collision_size_factor**0.85
    base = airtime_norm * congestion_scale * size_scale
    return _clip(base * (1.0 - success_ratio) ** 0.85, 0.0, 1.0)


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


def _apply_congestion_and_link_quality(
    *,
    node_windows: list[dict[str, object]],
    successes_by_node: dict[int, int],
    traffic_sent_by_node: dict[int, int],
    congestion_probability: float,
    link_success_min_ratio: float,
    rng: random.Random,
    round_id: int,
    snir_mode: str,
    snir_threshold_db: float,
    snir_threshold_min_db: float,
    snir_threshold_max_db: float,
    debug_step2: bool,
) -> tuple[dict[int, int], dict[str, float], float]:
    per_node_after_congestion: dict[int, int] = {}
    losses_congestion = 0
    for node_window in node_windows:
        node_id = int(node_window["node_id"])
        successes = successes_by_node.get(node_id, 0)
        successes_before = successes
        if successes > 0 and congestion_probability > 0.0:
            successes = sum(
                1 for _ in range(successes) if rng.random() > congestion_probability
            )
        per_node_after_congestion[node_id] = successes
        losses_congestion += successes_before - successes

    successes_after_congestion_total = sum(per_node_after_congestion.values())
    link_quality_weighted = 0.0
    if successes_after_congestion_total > 0:
        weighted_sum = 0.0
        for node_window in node_windows:
            node_id = int(node_window["node_id"])
            successes = per_node_after_congestion.get(node_id, 0)
            if successes <= 0:
                continue
            link_quality = float(node_window["link_quality"])
            weighted_sum += link_quality * successes
        link_quality_weighted = weighted_sum / successes_after_congestion_total
    snir_success_factor = _snir_success_factor(
        snir_mode=snir_mode,
        snir_threshold_db=snir_threshold_db,
        snir_threshold_min_db=snir_threshold_min_db,
        snir_threshold_max_db=snir_threshold_max_db,
    )
    link_quality_snir = _clip(link_quality_weighted * snir_success_factor, 0.0, 1.0)
    successes_after_link_total = sum(
        1
        for _ in range(successes_after_congestion_total)
        if rng.random() < link_quality_snir
    )
    min_ratio = _clip(link_success_min_ratio, 0.0, 1.0)
    if successes_after_congestion_total > 0:
        min_keep = max(1, int(round(successes_after_congestion_total * min_ratio)))
        if successes_after_link_total < min_keep:
            successes_after_link_total = min_keep
    if _should_debug_log(debug_step2, round_id):
        logger.debug(
            "SNIR facteur succès=%.3f (mode=%s seuil=%.2f dB min=%.2f dB max=%.2f dB "
            "link_quality_weighted=%.3f link_quality_snir=%.3f succès_après=%s).",
            snir_success_factor,
            snir_mode,
            snir_threshold_db,
            snir_threshold_min_db,
            snir_threshold_max_db,
            link_quality_weighted,
            link_quality_snir,
            successes_after_link_total,
        )
    logger.debug(
        "Diag congestion/lien (debug) - link_quality_weighted=%.3f successes_after_congestion_total=%s successes_after_link_total=%s",
        link_quality_weighted,
        successes_after_congestion_total,
        successes_after_link_total,
    )
    logger.info(
        "Diag congestion/lien - avant/after: congestion %s/%s lien %s/%s",
        sum(successes_by_node.values()),
        successes_after_congestion_total,
        successes_after_congestion_total,
        successes_after_link_total,
    )
    logger.info(
        "Diag congestion/lien - successes_after_congestion_total=%s link_quality_weighted=%.3f successes_after_link_total=%s",
        successes_after_congestion_total,
        link_quality_weighted,
        successes_after_link_total,
    )

    per_node_after_link: dict[int, int] = {
        int(node_window["node_id"]): 0 for node_window in node_windows
    }
    if successes_after_congestion_total > 0 and successes_after_link_total > 0:
        population: list[int] = []
        for node_id, count in per_node_after_congestion.items():
            if count > 0:
                population.extend([node_id] * count)
        if successes_after_link_total > len(population):
            logger.warning(
                "Succès après lien (%s) > population (%s), clamp au plafond.",
                successes_after_link_total,
                len(population),
            )
            successes_after_link_total = len(population)
        for node_id in rng.sample(population, k=successes_after_link_total):
            per_node_after_link[node_id] += 1
    for node_id in list(per_node_after_link.keys()):
        max_allowed = min(
            per_node_after_congestion.get(node_id, 0),
            traffic_sent_by_node.get(node_id, 0),
        )
        if per_node_after_link[node_id] > max_allowed:
            logger.warning(
                "Dépassement succès après lien (node_id=%s successes=%s cap=%s).",
                node_id,
                per_node_after_link[node_id],
                max_allowed,
            )
        per_node_after_link[node_id] = min(per_node_after_link[node_id], max_allowed)
        assert per_node_after_link[node_id] <= max_allowed
    losses_link_quality = successes_after_congestion_total - successes_after_link_total
    return (
        per_node_after_link,
        {
            "successes_before_congestion": sum(successes_by_node.values()),
            "successes_after_congestion": successes_after_congestion_total,
            "successes_before_link": successes_after_congestion_total,
            "successes_after_link": successes_after_link_total,
            "losses_congestion": losses_congestion,
            "losses_link_quality": losses_link_quality,
            "snir_success_factor": snir_success_factor,
            "link_quality_snir": link_quality_snir,
        },
        link_quality_weighted,
    )


def _snir_success_factor(
    *,
    snir_mode: str,
    snir_threshold_db: float,
    snir_threshold_min_db: float,
    snir_threshold_max_db: float,
) -> float:
    if snir_mode != "snir_on":
        return 1.0
    min_db = min(snir_threshold_min_db, snir_threshold_max_db)
    max_db = max(snir_threshold_min_db, snir_threshold_max_db)
    if max_db <= min_db:
        return 1.0
    threshold_db = _clamp_range(snir_threshold_db, min_db, max_db)
    threshold_ratio = (threshold_db - min_db) / max(max_db - min_db, 1e-6)
    softened_ratio = threshold_ratio**1.2
    attenuation = 1.0 - softened_ratio
    min_factor = 0.15
    return _clip(max(attenuation, min_factor), 0.0, 1.0)


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
    lambda_collision: float | None = DEFAULT_CONFIG.rl.lambda_collision,
    epsilon_greedy: float = 0.03,
    reward_floor: float | None = None,
    floor_on_zero_success: bool | None = None,
    density: float | None = None,
    snir_mode: str = "snir_on",
    snir_threshold_db: float | None = None,
    snir_threshold_min_db: float | None = None,
    snir_threshold_max_db: float | None = None,
    seed: int = 42,
    traffic_mode: str | None = None,
    jitter_range_s: float | None = None,
    window_duration_s: float | None = None,
    traffic_coeff_min: float | None = None,
    traffic_coeff_max: float | None = None,
    traffic_coeff_enabled: bool | None = None,
    traffic_coeff_scale: float | None = None,
    capture_probability: float | None = None,
    congestion_coeff: float | None = None,
    congestion_coeff_base: float | None = None,
    congestion_coeff_growth: float | None = None,
    congestion_coeff_max: float | None = None,
    link_success_min_ratio: float | None = None,
    network_load_min: float | None = None,
    network_load_max: float | None = None,
    collision_size_min: float | None = None,
    collision_size_under_max: float | None = None,
    collision_size_over_max: float | None = None,
    collision_size_factor: float | None = None,
    max_penalty_ratio: float | None = None,
    traffic_coeff_clamp_min: float | None = None,
    traffic_coeff_clamp_max: float | None = None,
    traffic_coeff_clamp_enabled: bool | None = None,
    window_delay_enabled: bool | None = None,
    window_delay_range_s: float | None = None,
    shadowing_sigma_db: float | None = None,
    reference_network_size: int | None = None,
    output_dir: Path | None = None,
    debug_step2: bool = False,
    reward_alert_level: str = "INFO",
    safe_profile: bool = False,
    no_clamp: bool = False,
    auto_collision_control: bool = False,
) -> Step2Result:
    """Exécute une simulation proxy de l'étape 2."""
    global _NO_CLAMP
    previous_no_clamp = _NO_CLAMP
    _NO_CLAMP = bool(no_clamp)
    rng = random.Random(seed)
    step2_defaults = DEFAULT_CONFIG.step2
    snir_defaults = DEFAULT_CONFIG.snir
    reward_floor_value = reward_floor
    if safe_profile and reward_floor_value is None:
        reward_floor_value = STEP2_SAFE_CONFIG.reward_floor
    traffic_mode_value = "poisson" if traffic_mode is None else traffic_mode
    reward_alert_level_value = (
        logging.WARNING if str(reward_alert_level).upper() == "WARNING" else logging.INFO
    )
    snir_threshold_value = (
        snir_defaults.snir_threshold_db
        if snir_threshold_db is None
        else float(snir_threshold_db)
    )
    snir_threshold_min_value = (
        snir_defaults.snir_threshold_min_db
        if snir_threshold_min_db is None
        else float(snir_threshold_min_db)
    )
    snir_threshold_max_value = (
        snir_defaults.snir_threshold_max_db
        if snir_threshold_max_db is None
        else float(snir_threshold_max_db)
    )
    if snir_threshold_min_value > snir_threshold_max_value:
        snir_threshold_min_value, snir_threshold_max_value = (
            snir_threshold_max_value,
            snir_threshold_min_value,
        )
    snir_meta = {
        "snir_threshold_db": snir_threshold_value,
        "snir_threshold_min_db": snir_threshold_min_value,
        "snir_threshold_max_db": snir_threshold_max_value,
    }
    jitter_range_value = jitter_range_s
    window_duration_value = (
        step2_defaults.window_duration_s if window_duration_s is None else window_duration_s
    )
    if window_size <= 0:
        raise ValueError("window_size doit être strictement positif.")
    if window_duration_value <= 0.0:
        raise ValueError("window_duration_s doit être strictement positif.")
    if jitter_range_value is not None and jitter_range_value < 0:
        logger.warning("jitter_range_s négatif (%.3f), forcé à 0.", jitter_range_value)
        jitter_range_value = 0.0
    tx_window_safety_factor = 1.2
    traffic_coeff_min_value = (
        step2_defaults.traffic_coeff_min if traffic_coeff_min is None else traffic_coeff_min
    )
    traffic_coeff_max_value = (
        step2_defaults.traffic_coeff_max if traffic_coeff_max is None else traffic_coeff_max
    )
    if traffic_coeff_min_value <= 0.0 or traffic_coeff_max_value <= 0.0:
        raise ValueError("traffic_coeff_min/max doivent être strictement positifs.")
    if traffic_coeff_min_value > traffic_coeff_max_value:
        traffic_coeff_min_value, traffic_coeff_max_value = (
            traffic_coeff_max_value,
            traffic_coeff_min_value,
        )
    traffic_coeff_scale_value = (
        step2_defaults.traffic_coeff_scale
        if traffic_coeff_scale is None
        else float(traffic_coeff_scale)
    )
    if traffic_coeff_scale_value <= 0.0:
        raise ValueError("traffic_coeff_scale doit être strictement positif.")
    traffic_coeff_enabled_value = (
        step2_defaults.traffic_coeff_enabled
        if traffic_coeff_enabled is None
        else traffic_coeff_enabled
    )
    traffic_coeff_clamp_min_value = (
        step2_defaults.traffic_coeff_clamp_min
        if traffic_coeff_clamp_min is None
        else traffic_coeff_clamp_min
    )
    traffic_coeff_clamp_max_value = (
        step2_defaults.traffic_coeff_clamp_max
        if traffic_coeff_clamp_max is None
        else traffic_coeff_clamp_max
    )
    if traffic_coeff_clamp_min_value > traffic_coeff_clamp_max_value:
        traffic_coeff_clamp_min_value, traffic_coeff_clamp_max_value = (
            traffic_coeff_clamp_max_value,
            traffic_coeff_clamp_min_value,
        )
    traffic_coeff_clamp_enabled_value = (
        step2_defaults.traffic_coeff_clamp_enabled
        if traffic_coeff_clamp_enabled is None
        else traffic_coeff_clamp_enabled
    )
    if traffic_coeff_clamp_enabled is True and not step2_defaults.traffic_coeff_clamp_enabled:
        logger.warning(
            "Le clamp des coefficients de trafic a été réactivé par l'utilisateur."
        )
    if no_clamp:
        traffic_coeff_clamp_enabled_value = False
    if floor_on_zero_success is None:
        if debug_step2:
            floor_on_zero_success_value = True
        else:
            floor_on_zero_success_value = (
                STEP2_SAFE_CONFIG.floor_on_zero_success
                if safe_profile
                else step2_defaults.floor_on_zero_success
            )
    else:
        floor_on_zero_success_value = floor_on_zero_success
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
    capture_probability_value = (
        step2_defaults.capture_probability
        if capture_probability is None
        else float(capture_probability)
    )
    capture_probability_value = _clip(capture_probability_value, 0.0, 1.0)
    congestion_coeff_value = (
        step2_defaults.congestion_coeff
        if congestion_coeff is None
        else float(congestion_coeff)
    )
    if congestion_coeff_value < 0.0:
        raise ValueError("congestion_coeff doit être positif.")
    congestion_coeff_base_value = (
        step2_defaults.congestion_coeff_base
        if congestion_coeff_base is None
        else float(congestion_coeff_base)
    )
    congestion_coeff_growth_value = (
        step2_defaults.congestion_coeff_growth
        if congestion_coeff_growth is None
        else float(congestion_coeff_growth)
    )
    congestion_coeff_max_value = (
        step2_defaults.congestion_coeff_max
        if congestion_coeff_max is None
        else float(congestion_coeff_max)
    )
    link_success_min_ratio_value = (
        step2_defaults.link_success_min_ratio
        if link_success_min_ratio is None
        else float(link_success_min_ratio)
    )
    if congestion_coeff_base_value < 0.0:
        raise ValueError("congestion_coeff_base doit être positif.")
    if congestion_coeff_growth_value <= 0.0:
        raise ValueError("congestion_coeff_growth doit être strictement positif.")
    if congestion_coeff_max_value < 0.0:
        raise ValueError("congestion_coeff_max doit être positif.")
    congestion_coeff_max_value = _clip(congestion_coeff_max_value, 0.0, 1.0)
    epsilon_greedy = _clip(epsilon_greedy, 0.0, 1.0)
    if density is not None:
        if int(density) != n_nodes:
            raise ValueError(
                "n_nodes doit correspondre à network_size (density) pour l'étape 2."
            )
        n_nodes = int(density)
    network_size_value = n_nodes
    if n_nodes <= 0:
        if n_nodes == 0:
            logger.error("network_size == 0 avant écriture des résultats.")
        raise ValueError("network_size doit être strictement positif.")
    pilot_network_size = min(DEFAULT_CONFIG.scenario.network_sizes)
    log_reward_components = n_nodes == pilot_network_size
    scenario_radius_m = float(DEFAULT_CONFIG.scenario.radius_m)
    if scenario_radius_m <= 0.0:
        raise ValueError("radius_m doit être strictement positif pour calculer la densité.")
    density_value = n_nodes / (math.pi * scenario_radius_m**2)
    algo_label = _algo_label(algorithm)
    raw_rows: list[dict[str, object]] = []
    selection_prob_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    all_link_qualities: list[float] = []
    all_snir_success_factors: list[float] = []
    all_link_quality_snir: list[float] = []
    node_clusters = assign_clusters(n_nodes, rng=rng)
    reference_size = (
        _default_reference_size()
        if reference_network_size is None
        else max(1, int(reference_network_size))
    )
    overload_ratio = max(0.0, (n_nodes / reference_size) - 1.0)
    if overload_ratio > 0.0:
        capture_probability_value = _clip(
            capture_probability_value / (1.0 + 0.75 * overload_ratio), 0.0, 1.0
        )
        congestion_coeff_base_value *= 1.0 + min(0.6, 0.45 * overload_ratio)
        congestion_coeff_growth_value *= 1.0 + min(0.6, 0.45 * overload_ratio)
        congestion_coeff_max_value = _clip(
            congestion_coeff_max_value * (1.0 + min(0.6, 0.45 * overload_ratio)),
            0.0,
            1.0,
        )
        if debug_step2:
            logger.info(
                "Ajustement congestion/capture (taille=%s ref=%s overload=%.2f): "
                "capture=%.3f base=%.3f growth=%.3f max=%.3f",
                n_nodes,
                reference_size,
                overload_ratio,
                capture_probability_value,
                congestion_coeff_base_value,
                congestion_coeff_growth_value,
                congestion_coeff_max_value,
            )
    load_clamp_min_value, load_clamp_max_value = _resolve_load_clamps(
        step2_defaults,
        network_load_min,
        network_load_max,
        safe_profile=bool(safe_profile),
        no_clamp=no_clamp,
    )
    (
        collision_clamp_min_value,
        collision_clamp_under_max_value,
        collision_clamp_over_max_value,
    ) = _resolve_collision_clamps(
        step2_defaults,
        collision_size_min,
        collision_size_under_max,
        collision_size_over_max,
        safe_profile=bool(safe_profile),
        no_clamp=no_clamp,
    )
    load_factor = _network_load_factor(
        n_nodes, reference_size, load_clamp_min_value, load_clamp_max_value
    )
    legacy_load_factor = _network_load_factor(n_nodes, reference_size, 0.6, 2.6)
    n_channels = max(1, len(DEFAULT_CONFIG.radio.channels_hz))
    congestion_probability = _congestion_collision_probability(
        n_nodes,
        reference_size,
        base_coeff=congestion_coeff_base_value,
        growth_coeff=congestion_coeff_growth_value,
        max_probability=congestion_coeff_max_value,
    )
    if congestion_coeff_value != 1.0:
        congestion_probability = _clip(
            congestion_probability * congestion_coeff_value, 0.0, 1.0
        )
    qos_clusters = tuple(DEFAULT_CONFIG.qos.clusters)
    traffic_size_factor = _traffic_coeff_size_factor(n_nodes, reference_size)
    traffic_variance_factor = _traffic_coeff_variance_factor(n_nodes, reference_size)
    traffic_coeff_midpoint = (traffic_coeff_min_value + traffic_coeff_max_value) / 2.0
    traffic_coeff_half_range = (traffic_coeff_max_value - traffic_coeff_min_value) / 2.0
    traffic_coeff_min_scaled = max(
        0.1, traffic_coeff_midpoint - traffic_coeff_half_range * traffic_variance_factor
    )
    traffic_coeff_max_scaled = traffic_coeff_midpoint + (
        traffic_coeff_half_range * traffic_variance_factor
    )
    if traffic_coeff_min_scaled > traffic_coeff_max_scaled:
        traffic_coeff_min_scaled, traffic_coeff_max_scaled = (
            traffic_coeff_max_scaled,
            traffic_coeff_min_scaled,
        )
    if debug_step2:
        logger.info(
            "Clamp traffic coeffs: enabled=%s min=%.3f max=%.3f",
            traffic_coeff_clamp_enabled_value,
            traffic_coeff_clamp_min_value,
            traffic_coeff_clamp_max_value,
        )
    traffic_coeffs = []
    traffic_coeffs_raw: list[float] = []
    traffic_coeff_clamp_flags: list[bool] = []
    for node_id in range(n_nodes):
        traffic_value = (
            rng.uniform(traffic_coeff_min_scaled, traffic_coeff_max_scaled)
            if traffic_coeff_enabled_value
            else 1.0
        )
        traffic_value *= (
            traffic_size_factor * _cluster_traffic_factor(node_clusters[node_id], qos_clusters)
        )
        traffic_coeffs_raw.append(traffic_value)
        if traffic_coeff_clamp_enabled_value:
            traffic_coeff_clamp_flags.append(
                traffic_value < traffic_coeff_clamp_min_value
                or traffic_value > traffic_coeff_clamp_max_value
            )
        else:
            traffic_coeff_clamp_flags.append(False)
    if traffic_coeff_clamp_enabled_value:
        clamped_count = sum(1 for flag in traffic_coeff_clamp_flags if flag)
        clamp_share = clamped_count / max(len(traffic_coeff_clamp_flags), 1)
        if clamp_share >= 0.8:
            logger.warning(
                "Clamping traffic_coeff ignoré (noeuds_clampes=%s/%s %.1f%%).",
                clamped_count,
                len(traffic_coeff_clamp_flags),
                clamp_share * 100.0,
            )
            traffic_coeffs = list(traffic_coeffs_raw)
        else:
            traffic_coeffs = [
                _clamp_range(
                    traffic_value,
                    traffic_coeff_clamp_min_value,
                    traffic_coeff_clamp_max_value,
                )
                for traffic_value in traffic_coeffs_raw
            ]
    else:
        traffic_coeffs = list(traffic_coeffs_raw)
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
    shadowing_sigma_base = (
        rng.uniform(6.0, 8.0) if shadowing_sigma_db is None else shadowing_sigma_db
    )
    shadowing_sigma_factor = _shadowing_sigma_size_factor(n_nodes, reference_size)
    base_shadowing_sigma_db = _clamp_range(
        shadowing_sigma_base * shadowing_sigma_factor, 4.0, 12.0
    )
    collision_size_factor_value = _collision_size_factor(
        n_nodes,
        reference_size,
        collision_clamp_min_value,
        collision_clamp_under_max_value,
        collision_clamp_over_max_value,
    )
    if collision_size_factor is not None:
        collision_size_factor_override = float(collision_size_factor)
        if collision_size_factor_override <= 0.0:
            raise ValueError("collision_size_factor doit être strictement positif.")
        if debug_step2:
            logger.info(
                "Override collision_size_factor: base=%.3f override=%.3f",
                collision_size_factor_value,
                collision_size_factor_override,
            )
        collision_size_factor_value = collision_size_factor_override
    legacy_collision_size_factor = _collision_size_factor(
        n_nodes, reference_size, 0.6, 1.0, 2.4
    )
    _log_size_factor_comparison(
        n_nodes,
        reference_size,
        load_factor=load_factor,
        collision_size_factor=collision_size_factor_value,
        legacy_load_factor=legacy_load_factor,
        legacy_collision_size_factor=legacy_collision_size_factor,
        load_clamp_min=load_clamp_min_value,
        load_clamp_max=load_clamp_max_value,
        collision_clamp_min=collision_clamp_min_value,
        collision_clamp_under_max=collision_clamp_under_max_value,
        collision_clamp_over_max=collision_clamp_over_max_value,
    )
    if lambda_collision is None:
        lambda_collision_base = step2_defaults.lambda_collision_base + 0.35 * lambda_energy
        size_scale = 1.0 + step2_defaults.lambda_collision_overload_scale * overload_ratio
        lambda_collision = _clip(
            lambda_collision_base * size_scale,
            step2_defaults.lambda_collision_min,
            step2_defaults.lambda_collision_max,
        )
    else:
        lambda_collision = _clip(lambda_collision, 0.0, 1.0)
    max_penalty_ratio_value = (
        step2_defaults.max_penalty_ratio
        if max_penalty_ratio is None
        else float(max_penalty_ratio)
    )
    if max_penalty_ratio_value < 0.0:
        raise ValueError("max_penalty_ratio doit être positif ou nul.")
    reward_weights = _reward_weights_for_algo(
        algorithm, reward_floor=reward_floor_value
    )
    sf_norm_by_sf = {
        sf: _normalize(sf, min(sf_values), max(sf_values)) for sf in sf_values
    }
    latency_norm_by_sf = energy_norm_by_sf
    reward_log_counts: dict[tuple[int, str, int], int] = {}
    reward_alert_state: dict[tuple[int, str], RewardAlertState] = {}
    reward_alert_global_counts: dict[tuple[int, str], int] = {}
    approx_threshold = 5000
    approx_sample_size = 2500
    approx_check_rounds = max(1, min(2, n_rounds))
    approx_discrepancy_threshold = 0.2
    approx_discrepancy_rounds = 0
    approx_adjusted = False

    if algorithm == "ucb1_sf":
        bandit = BanditUCB1(
            n_arms=n_arms,
            warmup_rounds=DEFAULT_CONFIG.rl.warmup,
            epsilon_min=0.02,
        )
        exploration_epsilon = max(epsilon_greedy, reward_weights.exploration_floor)
        window_start_s = 0.0
        for round_id in range(n_rounds):
            (
                collision_penalty_scale,
                reward_floor_boost,
            ) = _consume_reward_alert_adjustment(
                reward_alert_state, network_size_value, algo_label
            )
            effective_reward_weights = _apply_reward_floor_boost(
                reward_weights, reward_floor_boost
            )
            lambda_collision_effective = _clip(
                lambda_collision * collision_penalty_scale, 0.0, 1.0
            )
            _log_congestion_probability(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                congestion_probability=congestion_probability,
                congestion_coeff=congestion_coeff_value,
                congestion_coeff_base=congestion_coeff_base_value,
                congestion_coeff_growth=congestion_coeff_growth_value,
                congestion_coeff_max=congestion_coeff_max_value,
            )
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
            clamp_tracking_enabled = not no_clamp
            round_clamp_ratios: list[float] = []
            round_clamp_flags: list[bool] = []
            for node_id in range(n_nodes):
                sf_value = sf_values[arm_index]
                airtime_s = airtime_by_sf[sf_value]
                rate_multiplier = base_rate_multipliers[node_id]
                cluster = node_clusters[node_id]
                expected_sent_raw = max(
                    1,
                    int(
                        round(
                            window_size
                            * traffic_coeffs[node_id]
                            * rate_multiplier
                            * load_factor
                            * traffic_coeff_scale_value
                        )
                    ),
                )
                if _should_debug_log(debug_step2, round_id):
                    logger.debug(
                        "Traffic attendu avant plafonnement node=%s (sf=%s) expected_sent=%s.",
                        node_id,
                        sf_value,
                        expected_sent_raw,
                    )
                max_tx = expected_sent_raw
                if airtime_s > 0.0:
                    per_node_cap = max(
                        1, int(math.floor(window_duration_value / airtime_s))
                    )
                    window_cap = _max_window_tx(
                        window_duration_value,
                        airtime_s,
                        n_channels,
                        tx_window_safety_factor,
                    )
                    max_tx = min(max_tx, per_node_cap, window_cap)
                expected_sent = min(expected_sent_raw, max_tx)
                if clamp_tracking_enabled:
                    round_clamp_ratios.append(expected_sent / max(expected_sent_raw, 1))
                    round_clamp_flags.append(expected_sent < expected_sent_raw)
                if _should_debug_log(debug_step2, round_id):
                    logger.debug(
                        "Traffic attendu après plafonnement node=%s (sf=%s) expected_sent=%s (max_tx=%s).",
                        node_id,
                        sf_value,
                        expected_sent,
                        max_tx,
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
                if jitter_range_node_s <= 0.0:
                    traffic_times = _apply_phase_offset(
                        traffic_times,
                        rng=rng,
                        window_duration_s=window_duration_value,
                        base_period_s=base_period_s,
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
                link_quality = _apply_link_quality_variation(
                    rng,
                    _clip(shadowing_linear, 0.0, 1.0),
                    network_size=network_size_value,
                    reference_size=reference_size,
                )
                node_offset_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value and window_delay_range_value > 0
                    else 0.0
                )
                tx_starts = [window_start_s + node_offset_s + t for t in traffic_times]
                tx_channels = _assign_tx_channels(
                    tx_starts,
                    n_channels,
                    rng=rng,
                    mode="random",
                )
                effective_duration_s = _effective_window_duration(
                    tx_starts,
                    airtime_s,
                    window_duration_value,
                )
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
                        "tx_channels": tx_channels,
                        "effective_duration_s": effective_duration_s,
                        "shadowing_db": shadowing_db,
                        "shadowing_sigma_db": shadowing_sigma_db_node,
                        "link_quality": link_quality,
                    }
                )
            if clamp_tracking_enabled:
                (
                    traffic_coeff_scale_value,
                    window_duration_value,
                    tx_window_safety_factor,
                ) = _log_clamp_ratio_and_adjust(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    clamp_ratios=round_clamp_ratios,
                    clamp_flags=round_clamp_flags,
                    traffic_coeff_scale=traffic_coeff_scale_value,
                    window_duration_s=window_duration_value,
                    tx_window_safety_factor=tx_window_safety_factor,
                )
            (
                successes_by_node,
                traffic_sent_by_node,
                transmission_count,
                approx_collision_mode,
            ) = _compute_successes_and_traffic(
                node_windows,
                airtime_by_sf,
                rng=rng,
                capture_probability=capture_probability_value,
                approx_threshold=approx_threshold,
                approx_sample_size=approx_sample_size,
                debug_step2=debug_step2,
            )
            link_qualities = [
                float(node_window["link_quality"]) for node_window in node_windows
            ]
            all_link_qualities.extend(link_qualities)
            _log_pre_collision_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                traffic_sent_by_node=traffic_sent_by_node,
                link_qualities=link_qualities,
            )
            _log_link_quality_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                link_qualities=link_qualities,
            )
            _log_value_distribution(
                label="link_quality",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=link_qualities,
            )
            _log_pre_clip_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                successes_by_node=successes_by_node,
                traffic_sent_by_node=traffic_sent_by_node,
            )
            if approx_collision_mode:
                logger.debug(
                    "Mode approx collisions activé (%s transmissions).",
                    transmission_count,
                )
                approx_gap = _approx_collision_gap_ratio(
                    node_windows,
                    airtime_by_sf,
                    capture_probability=capture_probability_value,
                    approx_threshold=approx_threshold,
                    approx_sample_size=approx_sample_size,
                    seed=seed,
                    round_id=round_id,
                    max_rounds=approx_check_rounds,
                )
                if approx_gap is not None:
                    logger.info(
                        "Validation approx/exact (round=%s subset_nodes=%s) "
                        "ratio approx=%.4f exact=%.4f gap=%.3f.",
                        round_id,
                        int(approx_gap["subset_nodes"]),
                        approx_gap["approx_ratio"],
                        approx_gap["exact_ratio"],
                        approx_gap["gap_ratio"],
                    )
                    if approx_gap["gap_ratio"] >= approx_discrepancy_threshold:
                        approx_discrepancy_rounds += 1
            if (
                approx_collision_mode
                and not approx_adjusted
                and approx_discrepancy_rounds >= approx_check_rounds
            ):
                previous_threshold = approx_threshold
                previous_sample_size = approx_sample_size
                approx_threshold = max(1000, int(approx_threshold * 0.8))
                approx_sample_size = min(int(approx_sample_size * 1.5), 20000)
                approx_adjusted = True
                logger.warning(
                    "Ajustement approx collisions: threshold %s→%s sample_size %s→%s.",
                    previous_threshold,
                    approx_threshold,
                    previous_sample_size,
                    approx_sample_size,
                )
            collision_success_total = sum(successes_by_node.values())
            total_traffic_sent = sum(traffic_sent_by_node.values())
            if _should_debug_log(debug_step2, round_id):
                _log_link_quality_summary(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    link_qualities=link_qualities,
                )
                min_lq, median_lq, max_lq = _summarize_values(link_qualities)
                _log_debug_stage(
                    stage="avant_collisions",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "traffic_sent_total=%s tx_starts=%s "
                        "link_quality[min/med/max]=%.3f/%.3f/%.3f"
                    )
                    % (total_traffic_sent, transmission_count, min_lq, median_lq, max_lq),
                )
            if collision_success_total == 0:
                logger.warning(
                    "Aucun succès après collisions (taille=%s algo=%s round=%s).",
                    network_size_value,
                    algo_label,
                    round_id,
                )
            if _should_debug_log(debug_step2, round_id):
                min_succ, median_succ, max_succ = _summarize_values(
                    [float(value) for value in successes_by_node.values()]
                )
                _log_debug_stage(
                    stage="apres_collisions",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "collision_success_total=%s successes_by_node[min/med/max]=%.0f/%.1f/%.0f"
                    )
                    % (collision_success_total, min_succ, median_succ, max_succ),
                )
            per_node_successes, loss_stats, link_quality_weighted = (
                _apply_congestion_and_link_quality(
                    node_windows=node_windows,
                    successes_by_node=successes_by_node,
                    traffic_sent_by_node=traffic_sent_by_node,
                    congestion_probability=congestion_probability,
                    link_success_min_ratio=link_success_min_ratio_value,
                    rng=rng,
                    round_id=round_id,
                    snir_mode=snir_mode,
                    snir_threshold_db=snir_threshold_value,
                    snir_threshold_min_db=snir_threshold_min_value,
                    snir_threshold_max_db=snir_threshold_max_value,
                    debug_step2=debug_step2,
                )
            )
            snir_success_factor = float(loss_stats["snir_success_factor"])
            link_quality_snir = float(loss_stats["link_quality_snir"])
            all_snir_success_factors.append(snir_success_factor)
            all_link_quality_snir.append(link_quality_snir)
            _log_value_distribution(
                label="snir_success_factor",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=[snir_success_factor],
            )
            _log_value_distribution(
                label="link_quality_snir",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=[link_quality_snir],
            )
            _log_success_chain(
                network_size=network_size_value,
                n_nodes=n_nodes,
                algo_label=algo_label,
                round_id=round_id,
                total_traffic_sent=total_traffic_sent,
                successes_after_collisions=collision_success_total,
                successes_after_congestion=loss_stats["successes_after_congestion"],
                successes_after_link=loss_stats["successes_after_link"],
            )
            if _should_debug_log(debug_step2, round_id):
                _log_debug_stage(
                    stage="apres_congestion",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "successes_before_congestion=%s successes=%s"
                    )
                    % (
                        loss_stats["successes_before_congestion"],
                        loss_stats["successes_after_congestion"],
                    ),
                )
                _log_debug_stage(
                    stage="apres_link_quality",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "successes_before_link=%s successes=%s link_quality_weighted=%.3f"
                    )
                    % (
                        loss_stats["successes_before_link"],
                        loss_stats["successes_after_link"],
                        link_quality_weighted,
                    ),
                )
                _log_debug_stage(
                    stage="snir",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "snir_mode=%s snir_factor=%.3f link_quality_snir=%.3f"
                    )
                    % (
                        snir_mode,
                        loss_stats["snir_success_factor"],
                        loss_stats["link_quality_snir"],
                    ),
                )
            final_success_total = loss_stats["successes_after_link"]
            losses_congestion = loss_stats["losses_congestion"]
            losses_link_quality = loss_stats["losses_link_quality"]
            losses_collisions = max(total_traffic_sent - collision_success_total, 0)
            ratio_after_collisions = (
                collision_success_total / total_traffic_sent
                if total_traffic_sent > 0
                else 0.0
            )
            ratio_after_congestion = (
                loss_stats["successes_after_congestion"] / collision_success_total
                if collision_success_total > 0
                else 0.0
            )
            ratio_after_link = (
                loss_stats["successes_after_link"]
                / loss_stats["successes_after_congestion"]
                if loss_stats["successes_after_congestion"] > 0
                else 0.0
            )
            if debug_step2:
                _log_round_traffic_debug(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    traffic_sent_total=total_traffic_sent,
                    successes_total=final_success_total,
                )
            round_success_rates: list[float] = []
            round_rewards: list[float] = []
            round_collision_norms: list[float] = []
            round_throughput: list[float] = []
            round_energy_per_success: list[float] = []
            round_traffic_sent: list[int] = []
            round_successes: list[int] = []
            node_metrics: list[dict[str, object]] = []
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = traffic_sent_by_node.get(node_id, 0)
                successes = per_node_successes.get(node_id, 0)
                if successes > traffic_sent:
                    logger.warning(
                        "Clamp succès > trafic (node_id=%s successes=%s traffic_sent=%s).",
                        node_id,
                        successes,
                        traffic_sent,
                    )
                successes = min(successes, traffic_sent)
                success_flag = 1 if successes > 0 else 0
                failure_flag = 1 - success_flag
                airtime_norm = energy_norm_by_sf[sf_value]
                airtime_s = airtime_by_sf[sf_value]
                effective_duration_s = float(
                    node_window.get("effective_duration_s", window_duration_value)
                )
                collision_norm = _compute_collision_norm(
                    airtime_norm=airtime_norm,
                    congestion_probability=congestion_probability,
                    collision_size_factor=collision_size_factor_value,
                    successes=successes,
                    traffic_sent=traffic_sent,
                )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    airtime_norm,
                    collision_norm,
                    payload_bytes=payload_bytes,
                    effective_duration_s=effective_duration_s,
                    window_duration_s=window_duration_value,
                    airtime_s=airtime_s,
                )
                node_metrics.append(
                    {
                        "node_window": node_window,
                        "node_id": node_id,
                        "sf_value": sf_value,
                        "traffic_sent": traffic_sent,
                        "successes": successes,
                        "success_flag": success_flag,
                        "failure_flag": failure_flag,
                        "metrics": metrics,
                    }
                )
            throughput_values = [
                entry["metrics"].throughput_success for entry in node_metrics
            ]
            energy_success_values = [
                entry["metrics"].energy_per_success for entry in node_metrics
            ]
            throughput_min = min(throughput_values) if throughput_values else 0.0
            throughput_max = max(throughput_values) if throughput_values else 0.0
            energy_success_min = (
                min(energy_success_values) if energy_success_values else 0.0
            )
            energy_success_max = (
                max(energy_success_values) if energy_success_values else 0.0
            )
            for entry in node_metrics:
                node_window = entry["node_window"]
                node_id = int(entry["node_id"])
                sf_value = int(entry["sf_value"])
                traffic_sent = int(entry["traffic_sent"])
                successes = int(entry["successes"])
                success_flag = int(entry["success_flag"])
                failure_flag = int(entry["failure_flag"])
                metrics = entry["metrics"]
                throughput_norm = _normalize_local(
                    metrics.throughput_success, throughput_min, throughput_max
                )
                energy_per_success_norm = _normalize_local(
                    metrics.energy_per_success, energy_success_min, energy_success_max
                )
                log_components = (
                    log_reward_components and round_id == 0 and node_id == 0
                )
                log_context = (
                    f"pilot_size={network_size_value} algo={algo_label} "
                    f"round={round_id} node={node_id}"
                    if log_components
                    else None
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    traffic_sent,
                    sf_norm_by_sf[sf_value],
                    latency_norm_by_sf[sf_value],
                    metrics.energy_norm,
                    metrics.collision_norm,
                    throughput_norm,
                    energy_per_success_norm,
                    effective_reward_weights,
                    lambda_energy,
                    lambda_collision_effective,
                    max_penalty_ratio_value,
                    floor_on_zero_success=floor_on_zero_success_value,
                    log_components=log_components,
                    log_context=log_context,
                )
                window_rewards.append(reward)
                round_success_rates.append(metrics.success_rate)
                round_rewards.append(reward)
                round_collision_norms.append(metrics.collision_norm)
                round_throughput.append(metrics.throughput_success)
                round_energy_per_success.append(metrics.energy_per_success)
                round_traffic_sent.append(traffic_sent)
                round_successes.append(successes)
                common_raw_row = {
                    "network_size": network_size_value,
                    "density": density_value,
                    "algo": algo_label,
                    "snir_mode": snir_mode,
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
                    "success": success_flag,
                    "failure": failure_flag,
                    "success_rate": metrics.success_rate,
                    "bitrate_norm": metrics.bitrate_norm,
                    "energy_norm": metrics.energy_norm,
                    "collision_norm": metrics.collision_norm,
                    "throughput_success": metrics.throughput_success,
                    "energy_per_success": metrics.energy_per_success,
                    "ratio_successes_after_collisions": ratio_after_collisions,
                    "ratio_successes_after_congestion": ratio_after_congestion,
                    "ratio_successes_after_link": ratio_after_link,
                    "losses_collisions": losses_collisions,
                    "losses_congestion": losses_congestion,
                    "losses_link_quality": losses_link_quality,
                    "reward": reward,
                    **snir_meta,
                }
                raw_rows.append(
                    {
                        **common_raw_row,
                        "cluster": node_clusters[node_id],
                    }
                )
                raw_rows.append(
                    {
                        **common_raw_row,
                        "cluster": "all",
                    }
                )
            if auto_collision_control:
                traffic_coeff_scale_value, window_duration_value = _apply_collision_control(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    collision_norms=round_collision_norms,
                    traffic_coeff_scale=traffic_coeff_scale_value,
                    window_duration_s=window_duration_value,
                )
            _log_cluster_all_diagnostics(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                traffic_sent=round_traffic_sent,
                successes=round_successes,
                collision_norms=round_collision_norms,
                link_qualities=link_qualities,
                rewards=round_rewards,
                lambda_collision=lambda_collision,
                congestion_probability=congestion_probability,
                collision_size_factor=collision_size_factor_value,
            )
            _log_collision_stability(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                collision_norms=round_collision_norms,
                approx_collision_mode=approx_collision_mode,
            )
            _log_loss_breakdown(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                losses_collisions=losses_collisions,
                losses_congestion=losses_congestion,
                losses_link_quality=losses_link_quality,
            )
            _log_success_ratio_summary(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                total_traffic_sent=total_traffic_sent,
                ratio_after_collisions=ratio_after_collisions,
                ratio_after_congestion=ratio_after_congestion,
                ratio_after_link=ratio_after_link,
                losses_collisions=losses_collisions,
                losses_congestion=losses_congestion,
                losses_link_quality=losses_link_quality,
            )
            if collision_success_total > 0 and final_success_total == 0:
                avg_link_quality = (
                    sum(float(node_window["link_quality"]) for node_window in node_windows)
                    / len(node_windows)
                    if node_windows
                    else 0.0
                )
                logger.warning(
                    "Succès annulés après congestion/link_quality (taille=%s algo=%s round=%s "
                    "congestion=%.3f link_quality_avg=%.3f).",
                    network_size_value,
                    algo_label,
                    round_id,
                    congestion_probability,
                    avg_link_quality,
                )
            if _should_debug_log(debug_step2, round_id) and round_success_rates:
                avg_success = sum(round_success_rates) / len(round_success_rates)
                avg_reward = sum(round_rewards) / len(round_rewards)
                avg_collision = sum(round_collision_norms) / len(round_collision_norms)
                avg_throughput = sum(round_throughput) / len(round_throughput)
                avg_energy_per_success = (
                    sum(round_energy_per_success) / len(round_energy_per_success)
                )
                _log_debug_stage(
                    stage="final",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "success_rate=%.4f reward=%.4f collision_norm=%.4f "
                        "throughput_success=%.4f energy_per_success=%.4f"
                    )
                    % (
                        avg_success,
                        avg_reward,
                        avg_collision,
                        avg_throughput,
                        avg_energy_per_success,
                    ),
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
            _log_reward_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                rewards=window_rewards,
                reward_alert_level=reward_alert_level_value,
                log_counts=reward_log_counts,
                alert_state=reward_alert_state,
                global_alert_counts=reward_alert_global_counts,
                reward_floor=reward_weights.exploration_floor,
            )
            logger.info(
                "Round %s - %s : récompense moyenne = %.4f",
                round_id,
                algo_label,
                avg_reward,
            )
            learning_curve_rows.append(
                {
                    "network_size": network_size_value,
                    "density": density_value,
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
                        "density": density_value,
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
            (
                collision_penalty_scale,
                reward_floor_boost,
            ) = _consume_reward_alert_adjustment(
                reward_alert_state, network_size_value, algo_label
            )
            effective_reward_weights = _apply_reward_floor_boost(
                reward_weights, reward_floor_boost
            )
            lambda_collision_effective = _clip(
                lambda_collision * collision_penalty_scale, 0.0, 1.0
            )
            _log_congestion_probability(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                congestion_probability=congestion_probability,
                congestion_coeff=congestion_coeff_value,
                congestion_coeff_base=congestion_coeff_base_value,
                congestion_coeff_growth=congestion_coeff_growth_value,
                congestion_coeff_max=congestion_coeff_max_value,
            )
            window_rewards: list[float] = []
            if round_id > 0:
                delay_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value
                    else 0.0
                )
                window_start_s += window_duration_value + delay_s
            node_windows: list[dict[str, object]] = []
            clamp_tracking_enabled = not no_clamp
            round_clamp_ratios: list[float] = []
            round_clamp_flags: list[bool] = []
            for node_id in range(n_nodes):
                rate_multiplier = base_rate_multipliers[node_id]
                cluster = node_clusters[node_id]
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
                link_quality = _apply_link_quality_variation(
                    rng,
                    _clip(shadowing_linear, 0.0, 1.0),
                    network_size=network_size_value,
                    reference_size=reference_size,
                )
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
                airtime_s = airtime_by_sf[sf_value]
                expected_sent_raw = max(
                    1,
                    int(
                        round(
                            window_size
                            * traffic_coeffs[node_id]
                            * rate_multiplier
                            * load_factor
                            * traffic_coeff_scale_value
                        )
                    ),
                )
                if _should_debug_log(debug_step2, round_id):
                    logger.debug(
                        "Traffic attendu avant plafonnement node=%s expected_sent=%s.",
                        node_id,
                        expected_sent_raw,
                    )
                max_tx = expected_sent_raw
                if airtime_s > 0.0:
                    per_node_cap = max(
                        1, int(math.floor(window_duration_value / airtime_s))
                    )
                    window_cap = _max_window_tx(
                        window_duration_value,
                        airtime_s,
                        n_channels,
                        tx_window_safety_factor,
                    )
                    max_tx = min(max_tx, per_node_cap, window_cap)
                expected_sent = min(expected_sent_raw, max_tx)
                if clamp_tracking_enabled:
                    round_clamp_ratios.append(expected_sent / max(expected_sent_raw, 1))
                    round_clamp_flags.append(expected_sent < expected_sent_raw)
                if _should_debug_log(debug_step2, round_id):
                    logger.debug(
                        "Traffic attendu après plafonnement node=%s expected_sent=%s (max_tx=%s).",
                        node_id,
                        expected_sent,
                        max_tx,
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
                if jitter_range_node_s <= 0.0:
                    traffic_times = _apply_phase_offset(
                        traffic_times,
                        rng=rng,
                        window_duration_s=window_duration_value,
                        base_period_s=base_period_s,
                    )
                node_offset_s = (
                    rng.uniform(0.0, window_delay_range_value)
                    if window_delay_enabled_value and window_delay_range_value > 0
                    else 0.0
                )
                tx_starts = [window_start_s + node_offset_s + t for t in traffic_times]
                tx_channels = _assign_tx_channels(
                    tx_starts,
                    n_channels,
                    rng=rng,
                    mode="random",
                )
                effective_duration_s = _effective_window_duration(
                    tx_starts,
                    airtime_s,
                    window_duration_value,
                )
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
                        "tx_channels": tx_channels,
                        "effective_duration_s": effective_duration_s,
                        "shadowing_db": shadowing_db,
                        "shadowing_sigma_db": shadowing_sigma_db_node,
                        "link_quality": link_quality,
                    }
                )
            if clamp_tracking_enabled:
                (
                    traffic_coeff_scale_value,
                    window_duration_value,
                    tx_window_safety_factor,
                ) = _log_clamp_ratio_and_adjust(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    clamp_ratios=round_clamp_ratios,
                    clamp_flags=round_clamp_flags,
                    traffic_coeff_scale=traffic_coeff_scale_value,
                    window_duration_s=window_duration_value,
                    tx_window_safety_factor=tx_window_safety_factor,
                )
            (
                successes_by_node,
                traffic_sent_by_node,
                transmission_count,
                approx_collision_mode,
            ) = _compute_successes_and_traffic(
                node_windows,
                airtime_by_sf,
                rng=rng,
                capture_probability=capture_probability_value,
                approx_threshold=approx_threshold,
                approx_sample_size=approx_sample_size,
                debug_step2=debug_step2,
            )
            link_qualities = [
                float(node_window["link_quality"]) for node_window in node_windows
            ]
            all_link_qualities.extend(link_qualities)
            _log_pre_collision_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                traffic_sent_by_node=traffic_sent_by_node,
                link_qualities=link_qualities,
            )
            _log_link_quality_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                link_qualities=link_qualities,
            )
            _log_value_distribution(
                label="link_quality",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=link_qualities,
            )
            _log_pre_clip_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                successes_by_node=successes_by_node,
                traffic_sent_by_node=traffic_sent_by_node,
            )
            if approx_collision_mode:
                logger.debug(
                    "Mode approx collisions activé (%s transmissions).",
                    transmission_count,
                )
                approx_gap = _approx_collision_gap_ratio(
                    node_windows,
                    airtime_by_sf,
                    capture_probability=capture_probability_value,
                    approx_threshold=approx_threshold,
                    approx_sample_size=approx_sample_size,
                    seed=seed,
                    round_id=round_id,
                    max_rounds=approx_check_rounds,
                )
                if approx_gap is not None:
                    logger.info(
                        "Validation approx/exact (round=%s subset_nodes=%s) "
                        "ratio approx=%.4f exact=%.4f gap=%.3f.",
                        round_id,
                        int(approx_gap["subset_nodes"]),
                        approx_gap["approx_ratio"],
                        approx_gap["exact_ratio"],
                        approx_gap["gap_ratio"],
                    )
                    if approx_gap["gap_ratio"] >= approx_discrepancy_threshold:
                        approx_discrepancy_rounds += 1
            if (
                approx_collision_mode
                and not approx_adjusted
                and approx_discrepancy_rounds >= approx_check_rounds
            ):
                previous_threshold = approx_threshold
                previous_sample_size = approx_sample_size
                approx_threshold = max(1000, int(approx_threshold * 0.8))
                approx_sample_size = min(int(approx_sample_size * 1.5), 20000)
                approx_adjusted = True
                logger.warning(
                    "Ajustement approx collisions: threshold %s→%s sample_size %s→%s.",
                    previous_threshold,
                    approx_threshold,
                    previous_sample_size,
                    approx_sample_size,
                )
            collision_success_total = sum(successes_by_node.values())
            total_traffic_sent = sum(traffic_sent_by_node.values())
            if _should_debug_log(debug_step2, round_id):
                _log_link_quality_summary(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    link_qualities=link_qualities,
                )
                min_lq, median_lq, max_lq = _summarize_values(link_qualities)
                _log_debug_stage(
                    stage="avant_collisions",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "traffic_sent_total=%s tx_starts=%s "
                        "link_quality[min/med/max]=%.3f/%.3f/%.3f"
                    )
                    % (total_traffic_sent, transmission_count, min_lq, median_lq, max_lq),
                )
            if collision_success_total == 0:
                logger.warning(
                    "Aucun succès après collisions (taille=%s algo=%s round=%s).",
                    network_size_value,
                    algo_label,
                    round_id,
                )
            if _should_debug_log(debug_step2, round_id):
                min_succ, median_succ, max_succ = _summarize_values(
                    [float(value) for value in successes_by_node.values()]
                )
                _log_debug_stage(
                    stage="apres_collisions",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "collision_success_total=%s successes_by_node[min/med/max]=%.0f/%.1f/%.0f"
                    )
                    % (collision_success_total, min_succ, median_succ, max_succ),
                )
            per_node_successes, loss_stats, link_quality_weighted = (
                _apply_congestion_and_link_quality(
                    node_windows=node_windows,
                    successes_by_node=successes_by_node,
                    traffic_sent_by_node=traffic_sent_by_node,
                    congestion_probability=congestion_probability,
                    link_success_min_ratio=link_success_min_ratio_value,
                    rng=rng,
                    round_id=round_id,
                    snir_mode=snir_mode,
                    snir_threshold_db=snir_threshold_value,
                    snir_threshold_min_db=snir_threshold_min_value,
                    snir_threshold_max_db=snir_threshold_max_value,
                    debug_step2=debug_step2,
                )
            )
            snir_success_factor = float(loss_stats["snir_success_factor"])
            link_quality_snir = float(loss_stats["link_quality_snir"])
            all_snir_success_factors.append(snir_success_factor)
            all_link_quality_snir.append(link_quality_snir)
            _log_value_distribution(
                label="snir_success_factor",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=[snir_success_factor],
            )
            _log_value_distribution(
                label="link_quality_snir",
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                values=[link_quality_snir],
            )
            _log_success_chain(
                network_size=network_size_value,
                n_nodes=n_nodes,
                algo_label=algo_label,
                round_id=round_id,
                total_traffic_sent=total_traffic_sent,
                successes_after_collisions=collision_success_total,
                successes_after_congestion=loss_stats["successes_after_congestion"],
                successes_after_link=loss_stats["successes_after_link"],
            )
            if _should_debug_log(debug_step2, round_id):
                _log_debug_stage(
                    stage="apres_congestion",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "successes_before_congestion=%s successes=%s"
                    )
                    % (
                        loss_stats["successes_before_congestion"],
                        loss_stats["successes_after_congestion"],
                    ),
                )
                _log_debug_stage(
                    stage="apres_link_quality",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "successes_before_link=%s successes=%s link_quality_weighted=%.3f"
                    )
                    % (
                        loss_stats["successes_before_link"],
                        loss_stats["successes_after_link"],
                        link_quality_weighted,
                    ),
                )
            final_success_total = loss_stats["successes_after_link"]
            losses_congestion = loss_stats["losses_congestion"]
            losses_link_quality = loss_stats["losses_link_quality"]
            losses_collisions = max(total_traffic_sent - collision_success_total, 0)
            ratio_after_collisions = (
                collision_success_total / total_traffic_sent
                if total_traffic_sent > 0
                else 0.0
            )
            ratio_after_congestion = (
                loss_stats["successes_after_congestion"] / collision_success_total
                if collision_success_total > 0
                else 0.0
            )
            ratio_after_link = (
                loss_stats["successes_after_link"]
                / loss_stats["successes_after_congestion"]
                if loss_stats["successes_after_congestion"] > 0
                else 0.0
            )
            if debug_step2:
                _log_round_traffic_debug(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    traffic_sent_total=total_traffic_sent,
                    successes_total=final_success_total,
                )
            round_success_rates: list[float] = []
            round_rewards: list[float] = []
            round_collision_norms: list[float] = []
            round_throughput: list[float] = []
            round_energy_per_success: list[float] = []
            round_traffic_sent: list[int] = []
            round_successes: list[int] = []
            node_metrics: list[dict[str, object]] = []
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = traffic_sent_by_node.get(node_id, 0)
                successes = per_node_successes.get(node_id, 0)
                if successes > traffic_sent:
                    logger.warning(
                        "Clamp succès > trafic (node_id=%s successes=%s traffic_sent=%s).",
                        node_id,
                        successes,
                        traffic_sent,
                    )
                successes = min(successes, traffic_sent)
                success_flag = 1 if successes > 0 else 0
                failure_flag = 1 - success_flag
                airtime_norm = energy_norm_by_sf[sf_value]
                airtime_s = airtime_by_sf[sf_value]
                effective_duration_s = float(
                    node_window.get("effective_duration_s", window_duration_value)
                )
                collision_norm = _compute_collision_norm(
                    airtime_norm=airtime_norm,
                    congestion_probability=congestion_probability,
                    collision_size_factor=collision_size_factor_value,
                    successes=successes,
                    traffic_sent=traffic_sent,
                )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    airtime_norm,
                    collision_norm,
                    payload_bytes=payload_bytes,
                    effective_duration_s=effective_duration_s,
                    window_duration_s=window_duration_value,
                    airtime_s=airtime_s,
                )
                node_metrics.append(
                    {
                        "node_window": node_window,
                        "node_id": node_id,
                        "sf_value": sf_value,
                        "traffic_sent": traffic_sent,
                        "successes": successes,
                        "success_flag": success_flag,
                        "failure_flag": failure_flag,
                        "metrics": metrics,
                    }
                )
            throughput_values = [
                entry["metrics"].throughput_success for entry in node_metrics
            ]
            energy_success_values = [
                entry["metrics"].energy_per_success for entry in node_metrics
            ]
            throughput_min = min(throughput_values) if throughput_values else 0.0
            throughput_max = max(throughput_values) if throughput_values else 0.0
            energy_success_min = (
                min(energy_success_values) if energy_success_values else 0.0
            )
            energy_success_max = (
                max(energy_success_values) if energy_success_values else 0.0
            )
            for entry in node_metrics:
                node_window = entry["node_window"]
                node_id = int(entry["node_id"])
                sf_value = int(entry["sf_value"])
                traffic_sent = int(entry["traffic_sent"])
                successes = int(entry["successes"])
                success_flag = int(entry["success_flag"])
                failure_flag = int(entry["failure_flag"])
                metrics = entry["metrics"]
                throughput_norm = _normalize_local(
                    metrics.throughput_success, throughput_min, throughput_max
                )
                energy_per_success_norm = _normalize_local(
                    metrics.energy_per_success, energy_success_min, energy_success_max
                )
                log_components = (
                    log_reward_components and round_id == 0 and node_id == 0
                )
                log_context = (
                    f"pilot_size={network_size_value} algo={algo_label} "
                    f"round={round_id} node={node_id}"
                    if log_components
                    else None
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    traffic_sent,
                    sf_norm_by_sf[sf_value],
                    latency_norm_by_sf[sf_value],
                    metrics.energy_norm,
                    metrics.collision_norm,
                    throughput_norm,
                    energy_per_success_norm,
                    effective_reward_weights,
                    lambda_energy,
                    lambda_collision_effective,
                    max_penalty_ratio_value,
                    floor_on_zero_success=floor_on_zero_success_value,
                    log_components=log_components,
                    log_context=log_context,
                )
                window_rewards.append(reward)
                round_success_rates.append(metrics.success_rate)
                round_rewards.append(reward)
                round_collision_norms.append(metrics.collision_norm)
                round_throughput.append(metrics.throughput_success)
                round_energy_per_success.append(metrics.energy_per_success)
                round_traffic_sent.append(traffic_sent)
                round_successes.append(successes)
                common_raw_row = {
                    "network_size": network_size_value,
                    "density": density_value,
                    "algo": algo_label,
                    "snir_mode": snir_mode,
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
                    "success": success_flag,
                    "failure": failure_flag,
                    "success_rate": metrics.success_rate,
                    "bitrate_norm": metrics.bitrate_norm,
                    "energy_norm": metrics.energy_norm,
                    "collision_norm": metrics.collision_norm,
                    "throughput_success": metrics.throughput_success,
                    "energy_per_success": metrics.energy_per_success,
                    "ratio_successes_after_collisions": ratio_after_collisions,
                    "ratio_successes_after_congestion": ratio_after_congestion,
                    "ratio_successes_after_link": ratio_after_link,
                    "losses_collisions": losses_collisions,
                    "losses_congestion": losses_congestion,
                    "losses_link_quality": losses_link_quality,
                    "reward": reward,
                    **snir_meta,
                }
                raw_rows.append(
                    {
                        **common_raw_row,
                        "cluster": node_clusters[node_id],
                    }
                )
                raw_rows.append(
                    {
                        **common_raw_row,
                        "cluster": "all",
                    }
                )
            if auto_collision_control:
                traffic_coeff_scale_value, window_duration_value = _apply_collision_control(
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    collision_norms=round_collision_norms,
                    traffic_coeff_scale=traffic_coeff_scale_value,
                    window_duration_s=window_duration_value,
                )
            _log_cluster_all_diagnostics(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                traffic_sent=round_traffic_sent,
                successes=round_successes,
                collision_norms=round_collision_norms,
                link_qualities=link_qualities,
                rewards=round_rewards,
                lambda_collision=lambda_collision,
                congestion_probability=congestion_probability,
                collision_size_factor=collision_size_factor_value,
            )
            _log_collision_stability(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                collision_norms=round_collision_norms,
                approx_collision_mode=approx_collision_mode,
            )
            _log_loss_breakdown(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                losses_collisions=losses_collisions,
                losses_congestion=losses_congestion,
                losses_link_quality=losses_link_quality,
            )
            _log_success_ratio_summary(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                total_traffic_sent=total_traffic_sent,
                ratio_after_collisions=ratio_after_collisions,
                ratio_after_congestion=ratio_after_congestion,
                ratio_after_link=ratio_after_link,
                losses_collisions=losses_collisions,
                losses_congestion=losses_congestion,
                losses_link_quality=losses_link_quality,
            )
            if collision_success_total > 0 and final_success_total == 0:
                avg_link_quality = (
                    sum(float(node_window["link_quality"]) for node_window in node_windows)
                    / len(node_windows)
                    if node_windows
                    else 0.0
                )
                logger.warning(
                    "Succès annulés après congestion/link_quality (taille=%s algo=%s round=%s "
                    "congestion=%.3f link_quality_avg=%.3f).",
                    network_size_value,
                    algo_label,
                    round_id,
                    congestion_probability,
                    avg_link_quality,
                )
            if _should_debug_log(debug_step2, round_id) and round_success_rates:
                avg_success = sum(round_success_rates) / len(round_success_rates)
                avg_reward = sum(round_rewards) / len(round_rewards)
                avg_collision = sum(round_collision_norms) / len(round_collision_norms)
                avg_throughput = sum(round_throughput) / len(round_throughput)
                avg_energy_per_success = (
                    sum(round_energy_per_success) / len(round_energy_per_success)
                )
                _log_debug_stage(
                    stage="final",
                    network_size=network_size_value,
                    algo_label=algo_label,
                    round_id=round_id,
                    details=(
                        "success_rate=%.4f reward=%.4f collision_norm=%.4f "
                        "throughput_success=%.4f energy_per_success=%.4f"
                    )
                    % (
                        avg_success,
                        avg_reward,
                        avg_collision,
                        avg_throughput,
                        avg_energy_per_success,
                    ),
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
            _log_reward_stats(
                network_size=network_size_value,
                algo_label=algo_label,
                round_id=round_id,
                rewards=window_rewards,
                reward_alert_level=reward_alert_level_value,
                log_counts=reward_log_counts,
                alert_state=reward_alert_state,
                global_alert_counts=reward_alert_global_counts,
                reward_floor=reward_weights.exploration_floor,
            )
            logger.info(
                "Round %s - %s : récompense moyenne = %.4f",
                round_id,
                algo_label,
                avg_reward,
            )
            learning_curve_rows.append(
                {
                    "network_size": network_size_value,
                    "density": density_value,
                    "round": round_id,
                    "algo": algo_label,
                    "avg_reward": avg_reward,
                }
            )
    else:
        raise ValueError("algorithm doit être adr, mixra_h, mixra_opt ou ucb1_sf.")

    _log_link_quality_size_summary(
        network_size=network_size_value,
        algo_label=algo_label,
        link_qualities=all_link_qualities,
    )
    _log_value_distribution(
        label="snir_success_factor",
        network_size=network_size_value,
        algo_label=algo_label,
        values=all_snir_success_factors,
    )
    _log_value_distribution(
        label="link_quality_snir",
        network_size=network_size_value,
        algo_label=algo_label,
        values=all_link_quality_snir,
    )

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

    result = Step2Result(
        raw_rows=raw_rows,
        selection_prob_rows=selection_prob_rows,
        learning_curve_rows=learning_curve_rows,
    )
    _NO_CLAMP = previous_no_clamp
    return result
