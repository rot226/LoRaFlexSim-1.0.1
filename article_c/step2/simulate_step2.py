"""Simulation de l'étape 2 (proxy UCB1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

from article_c.common.csv_io import write_simulation_results
from article_c.step2.bandit_ucb1 import BanditUCB1


@dataclass(frozen=True)
class WindowMetrics:
    success_rate: float
    bitrate_norm: float
    energy_norm: float


@dataclass(frozen=True)
class Step2Result:
    raw_rows: list[dict[str, object]]


def _clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _compute_reward(
    success_rate: float, bitrate_norm: float, energy_norm: float, lambda_energy: float
) -> float:
    reward = success_rate * bitrate_norm - lambda_energy * energy_norm
    return _clip(reward, 0.0, 1.0)


def _simulate_window_metrics(
    rng: random.Random, arm: int, n_arms: int, window_size: int
) -> WindowMetrics:
    base_success = 0.9 - 0.08 * arm
    success_prob = _clip(base_success + rng.uniform(-0.05, 0.05), 0.1, 0.95)
    successes = sum(rng.random() < success_prob for _ in range(window_size))
    success_rate = successes / window_size
    bitrate_norm = (arm + 1) / n_arms
    energy_norm = _clip(0.2 + 0.8 * bitrate_norm + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    return WindowMetrics(
        success_rate=success_rate,
        bitrate_norm=bitrate_norm,
        energy_norm=energy_norm,
    )


def run_simulation(
    n_windows: int = 20,
    n_nodes: int = 12,
    n_arms: int = 4,
    bandit_mode: Literal["per_node", "global"] = "per_node",
    window_size: int = 10,
    lambda_energy: float = 0.3,
    density: int | None = None,
    snir_mode: str = "proxy",
    seed: int = 42,
    output_dir: Path | None = None,
) -> Step2Result:
    """Exécute une simulation proxy de l'étape 2.

    Le bandit peut être instancié par noeud (per_node) ou partagé (global).
    Les récompenses sont calculées par fenêtre de taille window_size.
    """
    rng = random.Random(seed)
    density_value = density if density is not None else n_nodes
    algo_label = f"ucb1_{bandit_mode}"
    raw_rows: list[dict[str, object]] = []

    if bandit_mode == "global":
        bandit = BanditUCB1(n_arms=n_arms)
        for window in range(n_windows):
            arm = bandit.select_arm()
            window_rewards: list[float] = []
            for node_id in range(n_nodes):
                metrics = _simulate_window_metrics(rng, arm, n_arms, window_size)
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
                        "window": window,
                        "node_id": node_id,
                        "arm": arm,
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "reward": reward,
                    }
                )
            avg_reward = sum(window_rewards) / len(window_rewards)
            bandit.update(arm, avg_reward)
    elif bandit_mode == "per_node":
        bandits = [BanditUCB1(n_arms=n_arms) for _ in range(n_nodes)]
        for window in range(n_windows):
            for node_id in range(n_nodes):
                bandit = bandits[node_id]
                arm = bandit.select_arm()
                metrics = _simulate_window_metrics(rng, arm, n_arms, window_size)
                reward = _compute_reward(
                    metrics.success_rate,
                    metrics.bitrate_norm,
                    metrics.energy_norm,
                    lambda_energy,
                )
                bandit.update(arm, reward)
                raw_rows.append(
                    {
                        "density": density_value,
                        "algo": algo_label,
                        "snir_mode": snir_mode,
                        "window": window,
                        "node_id": node_id,
                        "arm": arm,
                        "success_rate": metrics.success_rate,
                        "bitrate_norm": metrics.bitrate_norm,
                        "energy_norm": metrics.energy_norm,
                        "reward": reward,
                    }
                )
    else:
        raise ValueError("bandit_mode doit être 'per_node' ou 'global'.")

    if output_dir is not None:
        write_simulation_results(output_dir, raw_rows)

    return Step2Result(raw_rows=raw_rows)
