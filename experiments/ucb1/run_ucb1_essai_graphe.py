"""Essai graphe pour UCB1 avec paramètres fixes.

Le script exécute des simulations pour un petit ensemble de paramètres et
exporte deux fichiers CSV (SNIR on/off) contenant les colonnes suivantes :
``reward_mean, der, pdr, collisions, snir_mean, regret``.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

from loraflexsim.launcher import Simulator  # noqa: E402


@dataclass
class RunMetrics:
    seed: int
    num_nodes: int
    packet_interval_s: float
    packets_per_node: int
    reward_mean: float
    der: float
    pdr: float
    collisions: int
    snir_mean: float
    regret: float
    snir_state: str
    use_snir: bool


NODE_COUNTS = [300, 600, 1000]
PACKET_INTERVALS = [300.0, 600.0]
PACKETS_PER_NODE = 3
SEEDS = [1, 2]
OUTPUT_DIR = ROOT / "essai_graphe" / "step2"


def _apply_snir_config(sim: Simulator, use_snir: bool) -> None:
    sim.use_snir = bool(use_snir)
    if hasattr(sim, "channel") and sim.channel is not None:
        sim.channel.use_snir = bool(use_snir)
    multichannel = getattr(sim, "multichannel", None)
    for channel in getattr(multichannel, "channels", []) or []:
        channel.use_snir = bool(use_snir)


def _safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _event_reward(event: dict, *, node) -> float:
    success = event.get("result") == "Success"
    selector = getattr(node, "sf_selector", None) if node else None
    if selector is None:
        return 1.0 if success else 0.0
    snir_db = _safe_float(event.get("snir_dB"))
    airtime = float(event.get("end_time", 0.0) or 0.0) - float(
        event.get("start_time", 0.0) or 0.0
    )
    return selector.reward_from_outcome(
        success,
        snir_db=snir_db,
        snir_threshold_db=None,
        marginal_snir_margin_db=getattr(node.channel, "marginal_snir_margin_db", None)
        if node
        else None,
        airtime_s=airtime,
        energy_j=event.get("energy_J"),
        collision=event.get("result") != "Success" and event.get("heard"),
        expected_der=None,
        local_der=getattr(node, "pdr", None) if node else None,
    )


def _collect_metrics(sim: Simulator, *, seed: int) -> RunMetrics:
    metrics = sim.get_metrics()
    attempts = int(metrics.get("tx_attempted", 0) or 0)
    delivered = int(metrics.get("delivered", 0) or 0)
    der = delivered / attempts if attempts > 0 else 0.0
    pdr = float(metrics.get("PDR", 0.0) or 0.0)
    collisions = int(metrics.get("collisions", 0) or 0)
    use_snir = bool(getattr(sim, "use_snir", False))
    snir_state = "snir-on" if use_snir else "snir-off"

    events = list(getattr(sim, "events_log", []) or [])
    snir_values = [
        _safe_float(event.get("snir_dB"))
        for event in events
        if _safe_float(event.get("snir_dB")) is not None
    ]
    snir_mean = sum(snir_values) / len(snir_values) if snir_values else 0.0

    node_map = {node.id: node for node in sim.nodes}
    reward_means: list[float] = []
    for node in sim.nodes:
        selector = getattr(node, "sf_selector", None)
        if selector and selector.bandit.total_rounds > 0:
            reward_means.extend(selector.bandit.reward_window_mean)
    if reward_means:
        reward_mean = sum(reward_means) / len(reward_means)
    else:
        rewards = [
            _event_reward(event, node=node_map.get(event.get("node_id")))
            for event in events
        ]
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0

    regrets: list[float] = []
    for node in sim.nodes:
        selector = getattr(node, "sf_selector", None)
        if selector and selector.bandit.total_rounds > 0:
            window_values = selector.bandit.reward_window_mean
            if window_values:
                mean_reward = sum(window_values) / len(window_values)
                regrets.append(max(window_values) - mean_reward)
    regret = sum(regrets) / len(regrets) if regrets else 0.0

    return RunMetrics(
        seed=seed,
        num_nodes=len(sim.nodes),
        packet_interval_s=float(getattr(sim, "packet_interval", 0.0) or 0.0),
        packets_per_node=int(getattr(sim, "packets_to_send", 0) or 0),
        reward_mean=reward_mean,
        der=der,
        pdr=pdr,
        collisions=collisions,
        snir_mean=snir_mean,
        regret=regret,
        snir_state=snir_state,
        use_snir=use_snir,
    )


def run_essai_graphe(
    node_counts: Iterable[int] = NODE_COUNTS,
    intervals: Iterable[float] = PACKET_INTERVALS,
    seeds: Iterable[int] = SEEDS,
    *,
    packets_per_node: int = PACKETS_PER_NODE,
    snir_fading_std: float | None = 1.5,
) -> None:
    for use_snir in (True, False):
        rows: list[RunMetrics] = []
        for num_nodes in node_counts:
            for packet_interval in intervals:
                for seed in seeds:
                    sim = Simulator(
                        num_nodes=num_nodes,
                        num_gateways=1,
                        area_size=2000.0,
                        transmission_mode="Random",
                        packet_interval=packet_interval,
                        first_packet_interval=packet_interval,
                        packets_to_send=packets_per_node,
                        adr_node=False,
                        adr_server=False,
                        seed=seed,
                        snir_fading_std=snir_fading_std,
                    )
                    _apply_snir_config(sim, use_snir)
                    for node in sim.nodes:
                        node.learning_method = "ucb1"
                    sim.run()
                    rows.append(_collect_metrics(sim, seed=seed))

        suffix = "snir-on" if use_snir else "snir-off"
        output_path = OUTPUT_DIR / f"ucb1_essai_graphe_{suffix}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "num_nodes",
                    "packet_interval_s",
                    "packets_per_node",
                    "seed",
                    "reward_mean",
                    "der",
                    "pdr",
                    "collisions",
                    "snir_mean",
                    "regret",
                    "snir_state",
                    "use_snir",
                ]
            )
            for entry in rows:
                writer.writerow(
                    [
                        entry.num_nodes,
                        f"{entry.packet_interval_s:.6f}",
                        entry.packets_per_node,
                        entry.seed,
                        f"{entry.reward_mean:.6f}",
                        f"{entry.der:.6f}",
                        f"{entry.pdr:.6f}",
                        entry.collisions,
                        f"{entry.snir_mean:.6f}",
                        f"{entry.regret:.6f}",
                        entry.snir_state,
                        entry.use_snir,
                    ]
                )


if __name__ == "__main__":
    run_essai_graphe()
