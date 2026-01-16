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


@dataclass(frozen=True)
class Step2Result:
    raw_rows: list[dict[str, object]]
    selection_prob_rows: list[dict[str, object]]
    learning_curve_rows: list[dict[str, object]]


logger = logging.getLogger(__name__)


def _clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _compute_reward(
    success_rate: float, bitrate_norm: float, energy_norm: float, lambda_energy: float
) -> float:
    reward = success_rate * bitrate_norm - lambda_energy * energy_norm
    return _clip(reward, 0.0, 1.0)


def _compute_window_metrics(
    successes: int, traffic_sent: int, bitrate_norm: float, energy_norm: float
) -> WindowMetrics:
    success_rate = successes / traffic_sent if traffic_sent > 0 else 0.0
    return WindowMetrics(
        success_rate=success_rate,
        bitrate_norm=bitrate_norm,
        energy_norm=energy_norm,
    )


def _compute_collision_successes(
    transmissions: dict[int, list[tuple[float, float, int]]]
) -> dict[int, int]:
    successes_by_node: dict[int, int] = {}
    for events in transmissions.values():
        if not events:
            continue
        collided = [False] * len(events)
        indexed_events = list(enumerate(events))
        indexed_events.sort(key=lambda item: item[1][0])
        active: list[tuple[float, int]] = []
        for event_index, (start, end, _node_id) in indexed_events:
            active = [
                (active_end, active_index)
                for active_end, active_index in active
                if active_end > start
            ]
            if active:
                collided[event_index] = True
                for _active_end, active_index in active:
                    collided[active_index] = True
            active.append((end, event_index))
        for event_index, (_start, _end, node_id) in enumerate(events):
            if not collided[event_index]:
                successes_by_node[node_id] = successes_by_node.get(node_id, 0) + 1
    return successes_by_node


def _weights_for_algo(algorithm: str, n_arms: int) -> list[float]:
    if algorithm == "mixra_h":
        base = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]
    elif algorithm == "mixra_opt":
        base = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
    else:
        base = [1.0] + [0.0] * 5
    weights = base[:n_arms]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


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
    window_size: int = 10,
    lambda_energy: float = 0.3,
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
    density_value = density if density is not None else n_nodes
    algo_label = _algo_label(algorithm)
    raw_rows: list[dict[str, object]] = []
    selection_prob_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    node_clusters = assign_clusters(n_nodes, rng=rng)
    traffic_coeffs = [
        rng.uniform(traffic_coeff_min_value, traffic_coeff_max_value)
        if traffic_coeff_enabled_value
        else 1.0
        for _ in range(n_nodes)
    ]

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

    if algorithm == "ucb1_sf":
        bandit = BanditUCB1(n_arms=n_arms)
        window_start_s = 0.0
        for round_id in range(n_rounds):
            arm_index = bandit.select_arm()
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
                rate_multiplier = rng.uniform(0.8, 1.2)
                expected_sent = max(
                    1, int(round(window_size * traffic_coeffs[node_id] * rate_multiplier))
                )
                base_period_s = window_duration_value / expected_sent
                jitter_range_node_s = (
                    0.3 * base_period_s
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
                shadowing_sigma_db = rng.uniform(4.0, 8.0)
                shadowing_db = rng.gauss(0.0, shadowing_sigma_db)
                link_quality = _clip(10 ** (-shadowing_db / 10.0), 0.0, 1.0)
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
                        "shadowing_sigma_db": shadowing_sigma_db,
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
            successes_by_node = _compute_collision_successes(transmissions_by_sf)
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = int(node_window["traffic_sent"])
                successes = successes_by_node.get(node_id, 0)
                link_quality = float(node_window["link_quality"])
                if link_quality < 1.0 and successes > 0:
                    successes = sum(
                        1 for _ in range(successes) if rng.random() < link_quality
                    )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    energy_norm_by_sf[sf_value],
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    metrics.bitrate_norm,
                    metrics.energy_norm,
                    lambda_energy,
                )
                window_rewards.append(reward)
                raw_rows.append(
                    {
                        "density": density_value,
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
                        "reward": reward,
                    }
                )
                raw_rows.append(
                    {
                        "density": density_value,
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
                        "round": round_id,
                        "sf": sf_value,
                        "selection_prob": bandit.counts[sf_index] / total,
                    }
                )
    elif algorithm in {"adr", "mixra_h", "mixra_opt"}:
        weights = _weights_for_algo(algorithm, n_arms)
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
                if algorithm == "adr":
                    arm_index = 0
                else:
                    arm_index = rng.choices(range(n_arms), weights=weights, k=1)[0]
                sf_value = sf_values[arm_index]
                rate_multiplier = rng.uniform(0.8, 1.2)
                expected_sent = max(
                    1, int(round(window_size * traffic_coeffs[node_id] * rate_multiplier))
                )
                base_period_s = window_duration_value / expected_sent
                jitter_range_node_s = (
                    0.3 * base_period_s
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
                shadowing_sigma_db = rng.uniform(4.0, 8.0)
                shadowing_db = rng.gauss(0.0, shadowing_sigma_db)
                link_quality = _clip(10 ** (-shadowing_db / 10.0), 0.0, 1.0)
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
                        "shadowing_sigma_db": shadowing_sigma_db,
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
            successes_by_node = _compute_collision_successes(transmissions_by_sf)
            for node_window in node_windows:
                node_id = int(node_window["node_id"])
                sf_value = int(node_window["sf"])
                traffic_sent = int(node_window["traffic_sent"])
                successes = successes_by_node.get(node_id, 0)
                link_quality = float(node_window["link_quality"])
                if link_quality < 1.0 and successes > 0:
                    successes = sum(
                        1 for _ in range(successes) if rng.random() < link_quality
                    )
                metrics = _compute_window_metrics(
                    successes,
                    traffic_sent,
                    bitrate_norm_by_sf[sf_value],
                    energy_norm_by_sf[sf_value],
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    metrics.bitrate_norm,
                    metrics.energy_norm,
                    lambda_energy,
                )
                window_rewards.append(reward)
                raw_rows.append(
                    {
                        "density": density_value,
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
                        "reward": reward,
                    }
                )
                raw_rows.append(
                    {
                        "density": density_value,
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
                    "round": round_id,
                    "algo": algo_label,
                    "avg_reward": avg_reward,
                }
            )
    else:
        raise ValueError("algorithm doit être adr, mixra_h, mixra_opt ou ucb1_sf.")

    if output_dir is not None:
        write_simulation_results(output_dir, raw_rows)
        learning_curve_path = output_dir / "learning_curve.csv"
        learning_curve_header = ["round", "algo", "avg_reward"]
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
