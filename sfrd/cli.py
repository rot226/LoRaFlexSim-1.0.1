"""Contrat d'appel stable pour lancer une campagne uplink LoRaFlexSim.

Ce module fournit la fonction ``run_campaign`` destinée à être appelée depuis
``sfrd.cli.run_campaign``. L'implémentation s'appuie directement sur les
objets internes LoRaFlexSim (``Simulator`` + ``QoSManager``) sans dupliquer la
logique de simulation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from loraflexsim.launcher import Channel, Simulator
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.learning import LoRaSFSelectorUCB1
from sfrd.parse.reward_ucb import collect_ucb_history, export_ucb_history_csv, learning_curve_from_history

_ALGORITHM_ALIASES = {
    "adr": "ADR-Pure",
    "adr-pure": "ADR-Pure",
    "apra": "APRA-like",
    "apra-like": "APRA-like",
    "aimi": "Aimi-like",
    "aimi-like": "Aimi-like",
    "mixra-opt": "MixRA-Opt",
    "mixra_h": "MixRA-H",
    "mixra-h": "MixRA-H",
    "ucb": "UCB1",
    "ucb1": "UCB1",
}

_SNIR_ALIASES = {
    "snir_on": True,
    "on": True,
    "true": True,
    "snir_off": False,
    "off": False,
    "false": False,
}


def _normalize_algorithm(value: str) -> str:
    normalized = value.strip().lower()
    return _ALGORITHM_ALIASES.get(normalized, value)


def _normalize_snir_mode(value: str) -> bool:
    key = value.strip().lower()
    if key not in _SNIR_ALIASES:
        raise ValueError(f"snir_mode invalide: {value!r}")
    return _SNIR_ALIASES[key]


def run_campaign(
    *,
    network_size: int,
    algorithm: str,
    snir_mode: str,
    seed: int,
    warmup_s: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Exécute une campagne uplink non-interactive avec 1 gateway.

    Point d'entrée interne invoqué explicitement:
    ``loraflexsim.launcher.Simulator.run``.
    """

    if network_size <= 0:
        raise ValueError("network_size doit être strictement positif")
    if warmup_s < 0 or not math.isfinite(warmup_s):
        raise ValueError("warmup_s doit être un flottant fini >= 0")

    resolved_algorithm = _normalize_algorithm(algorithm)
    use_snir = _normalize_snir_mode(snir_mode)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    packet_interval_s = 120.0
    warmup_intervals = int(math.ceil(warmup_s / packet_interval_s))

    channel = Channel(snir_model=use_snir, use_snir=use_snir, phy_model="omnet_full")
    simulator = Simulator(
        num_nodes=int(network_size),
        num_gateways=1,
        area_size=2000.0,
        transmission_mode="Random",
        packet_interval=packet_interval_s,
        first_packet_interval=packet_interval_s,
        warm_up_intervals=warmup_intervals,
        packets_to_send=6,
        duty_cycle=0.01,
        mobility=False,
        channels=[channel],
        seed=int(seed),
        payload_size_bytes=20,
        phy_model="omnet_full",
    )

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[1.0 / packet_interval_s],
        pdr_targets=[0.9],
    )
    if resolved_algorithm == "UCB1":
        manager.apply(simulator, "ADR-Pure", use_snir=use_snir)
        for node in simulator.nodes:
            node.adr = False
            node.learning_method = "ucb1"
            if getattr(node, "sf_selector", None) is None:
                node.sf_selector = LoRaSFSelectorUCB1(
                    success_weight=1.0,
                    snir_margin_weight=0.0,
                    energy_penalty_weight=0.5,
                    reward_mode="qos",
                )
    else:
        manager.apply(simulator, resolved_algorithm, use_snir=use_snir)

    # Fonction interne LoRaFlexSim réellement appelée pour exécuter la
    # simulation événementielle.
    simulator.run()
    metrics = simulator.get_metrics()

    ucb_history = collect_ucb_history(simulator)
    learning_curve = learning_curve_from_history(ucb_history)
    if ucb_history:
        export_ucb_history_csv(ucb_history, output_path / "ucb_history.csv")

    summary = {
        "contract": {
            "network_size": int(network_size),
            "algorithm": resolved_algorithm,
            "snir_mode": "snir_on" if use_snir else "snir_off",
            "seed": int(seed),
            "warmup_s": float(warmup_s),
            "output_dir": str(output_path),
        },
        "runtime": {
            "gateways": 1,
            "packets_to_send": 6,
            "packet_interval_s": packet_interval_s,
            "warm_up_intervals": warmup_intervals,
            "internal_entrypoint": "loraflexsim.launcher.Simulator.run",
        },
        "metrics": {
            "pdr": float(metrics.get("PDR", 0.0)),
            "throughput_bps": float(metrics.get("throughput_bps", 0.0)),
            "collisions": int(metrics.get("collisions", 0)),
            "tx_attempted": int(metrics.get("tx_attempted", 0)),
            "rx_delivered": int(metrics.get("rx_delivered", 0)),
            "ucb_learning_curve": learning_curve,
        },
    }

    summary_path = output_path / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "summary_path": summary_path,
        "summary": summary,
    }
