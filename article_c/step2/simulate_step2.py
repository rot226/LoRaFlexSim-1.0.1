"""Simulation de l'étape 2 (proxy UCB1 et comparaisons)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_simulation_results
from article_c.common.lora_phy import bitrate_lora, coding_rate_to_cr, compute_airtime
from article_c.common.utils import assign_clusters
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


def _clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _compute_reward(
    success_rate: float, bitrate_norm: float, energy_norm: float, lambda_energy: float
) -> float:
    reward = success_rate * bitrate_norm - lambda_energy * energy_norm
    return _clip(reward, 0.0, 1.0)


def _simulate_window_metrics(
    rng: random.Random,
    arm_index: int,
    window_size: int,
    bitrate_norm: float,
    energy_norm: float,
) -> WindowMetrics:
    base_success = 0.9 - 0.08 * arm_index
    success_prob = _clip(base_success + rng.uniform(-0.05, 0.05), 0.1, 0.95)
    successes = sum(rng.random() < success_prob for _ in range(window_size))
    success_rate = successes / window_size
    energy_norm = _clip(energy_norm + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    return WindowMetrics(
        success_rate=success_rate,
        bitrate_norm=bitrate_norm,
        energy_norm=energy_norm,
    )


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
    output_dir: Path | None = None,
) -> Step2Result:
    """Exécute une simulation proxy de l'étape 2."""
    rng = random.Random(seed)
    density_value = density if density is not None else n_nodes
    algo_label = _algo_label(algorithm)
    raw_rows: list[dict[str, object]] = []
    selection_prob_rows: list[dict[str, object]] = []
    node_clusters = assign_clusters(n_nodes, rng=rng)

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
        for round_id in range(n_rounds):
            arm_index = bandit.select_arm()
            window_rewards: list[float] = []
            for node_id in range(n_nodes):
                sf_value = sf_values[arm_index]
                metrics = _simulate_window_metrics(
                    rng,
                    arm_index,
                    window_size,
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
                        "sf": sf_values[arm_index],
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
                        "sf": sf_values[arm_index],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "reward": reward,
                    }
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
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
        for round_id in range(n_rounds):
            for node_id in range(n_nodes):
                if algorithm == "adr":
                    arm_index = 0
                else:
                    arm_index = rng.choices(range(n_arms), weights=weights, k=1)[0]
                sf_value = sf_values[arm_index]
                metrics = _simulate_window_metrics(
                    rng,
                    arm_index,
                    window_size,
                    bitrate_norm_by_sf[sf_value],
                    energy_norm_by_sf[sf_value],
                )
                reward = _compute_reward(
                    metrics.success_rate,
                    metrics.bitrate_norm,
                    metrics.energy_norm,
                    lambda_energy,
                )
                raw_rows.append(
                    {
                        "density": density_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "cluster": node_clusters[node_id],
                        "round": round_id,
                        "node_id": node_id,
                        "sf": sf_values[arm_index],
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
                        "sf": sf_values[arm_index],
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "reward": reward,
                    }
                )
    else:
        raise ValueError("algorithm doit être adr, mixra_h, mixra_opt ou ucb1_sf.")

    if output_dir is not None:
        write_simulation_results(output_dir, raw_rows)

    return Step2Result(raw_rows=raw_rows, selection_prob_rows=selection_prob_rows)
