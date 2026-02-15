"""Contrat d'appel stable pour lancer une campagne uplink LoRaFlexSim.

Ce module fournit la fonction ``run_campaign`` destinée à être appelée depuis
``sfrd.cli.run_campaign``. L'implémentation s'appuie directement sur les
objets internes LoRaFlexSim (``Simulator`` + ``QoSManager``) sans dupliquer la
logique de simulation.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loraflexsim.launcher import Channel, Simulator
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.learning import LoRaSFSelectorUCB1
from sfrd.parse.reward_ucb import (
    collect_ucb_history,
    export_ucb_history_csv,
    learning_curve_from_history,
    load_ucb_config,
)

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


def _export_raw_artifacts(simulator: Simulator, output_path: Path) -> None:
    """Exporte les artefacts bruts attendus (`raw_packets.csv`, `raw_energy.csv`)."""

    payload_bytes = int(getattr(simulator, "payload_size_bytes", 0) or 0)
    packet_rows: list[dict[str, int | float]] = []
    max_time = 0.0

    for event in getattr(simulator, "events_log", []):
        start_time = event.get("start_time")
        node_id = event.get("node_id")
        sf = event.get("sf")
        if start_time is None or node_id is None or sf is None:
            continue
        try:
            time_value = float(start_time)
            node_value = int(node_id)
            sf_value = int(sf)
        except (TypeError, ValueError):
            continue
        if sf_value < 7 or sf_value > 12:
            continue
        if time_value > max_time:
            max_time = time_value

        packet_rows.append(
            {
                "time": time_value,
                "node_id": node_value,
                "sf": sf_value,
                "tx_ok": 1,
                "rx_ok": 1 if str(event.get("result", "")).strip() == "Success" else 0,
                "payload_bytes": payload_bytes,
                "run": 1,
            }
        )

    with (output_path / "raw_packets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "node_id", "sf", "tx_ok", "rx_ok", "payload_bytes", "run"],
        )
        writer.writeheader()
        writer.writerows(packet_rows)

    total_energy_joule = float(sum(getattr(node, "energy_consumed", 0.0) for node in simulator.nodes))
    with (output_path / "raw_energy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["total_energy_joule", "sim_duration_s"])
        writer.writeheader()
        writer.writerow(
            {
                "total_energy_joule": total_energy_joule,
                "sim_duration_s": max_time,
            }
        )


def run_campaign(
    *,
    network_size: int,
    algorithm: str,
    snir_mode: str,
    seed: int,
    warmup_s: float,
    output_dir: str | Path,
    ucb_config_path: str | Path | None = None,
    heartbeat_callback: Callable[[dict[str, Any]], None] | None = None,
    heartbeat_interval_s: float = 45.0,
    max_run_seconds: float | None = None,
) -> dict[str, Any]:
    """Exécute une campagne uplink non-interactive avec 1 gateway.

    Point d'entrée interne invoqué explicitement:
    ``loraflexsim.launcher.Simulator.run``.
    """

    if network_size <= 0:
        raise ValueError("network_size doit être strictement positif")
    if warmup_s < 0 or not math.isfinite(warmup_s):
        raise ValueError("warmup_s doit être un flottant fini >= 0")
    if max_run_seconds is not None:
        if not isinstance(max_run_seconds, (float, int)):
            raise ValueError("max_run_seconds doit être un nombre réel > 0")
        max_run_seconds = float(max_run_seconds)
        if not math.isfinite(max_run_seconds) or max_run_seconds <= 0.0:
            raise ValueError("max_run_seconds doit être un flottant fini > 0")

    resolved_algorithm = _normalize_algorithm(algorithm)
    use_snir = _normalize_snir_mode(snir_mode)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ucb_config = load_ucb_config(ucb_config_path)

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
        ucb_selector_kwargs={
            "success_weight": 1.0,
            "snir_margin_weight": 0.0,
            "energy_penalty_weight": ucb_config.lambda_e,
            "reward_mode": "qos",
            "reward_window": ucb_config.reward_window,
            "exploration_coefficient": ucb_config.exploration_coefficient,
        },
        ucb_episode_mode=ucb_config.episode.mode,
        ucb_episode_packet_window=ucb_config.episode.packet_window,
        ucb_episode_time_window_s=ucb_config.episode.time_window_s,
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
                node.sf_selector = LoRaSFSelectorUCB1(**simulator.ucb_selector_kwargs)
    else:
        manager.apply(simulator, resolved_algorithm, use_snir=use_snir)

    # Fonction interne LoRaFlexSim réellement appelée pour exécuter la
    # simulation événementielle.
    run_error: Exception | None = None

    def _run_simulator() -> None:
        nonlocal run_error
        try:
            simulator.run()
        except Exception as exc:  # pragma: no cover - surface l'exception au thread principal
            run_error = exc

    runner = threading.Thread(target=_run_simulator, name="sfrd-sim-runner", daemon=True)
    runner.start()

    try:
        interval_s = float(heartbeat_interval_s)
    except (TypeError, ValueError):
        interval_s = 45.0
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        interval_s = 45.0

    wall_clock_start = time.perf_counter()
    timed_out = False
    while runner.is_alive():
        runner.join(timeout=interval_s)
        elapsed_s = time.perf_counter() - wall_clock_start
        if max_run_seconds is not None and elapsed_s >= max_run_seconds:
            simulator.stop()
            runner.join(timeout=min(interval_s, 1.0))
            timed_out = True
            break
        if heartbeat_callback is None or not runner.is_alive():
            continue
        heartbeat_callback(
            {
                "sim_time_s": float(getattr(simulator, "current_time", 0.0) or 0.0),
                "events_processed": int(getattr(simulator, "events_processed", 0) or 0),
                "last_qos_refresh_sim_time": getattr(simulator, "last_qos_refresh_sim_time", None),
                "qos_refresh_count": int(getattr(simulator, "_qos_refresh_count", 0) or 0),
                "timestamp": time.time(),
            }
        )

    if run_error is not None:
        raise run_error
    if timed_out:
        raise TimeoutError(
            (
                "Durée limite run dépassée "
                f"({elapsed_s:.1f}s >= {max_run_seconds:.1f}s)."
            )
        )

    metrics = simulator.get_metrics()
    _export_raw_artifacts(simulator, output_path)

    ucb_history = collect_ucb_history(simulator)
    learning_curve = learning_curve_from_history(ucb_history)
    if ucb_history:
        export_ucb_history_csv(ucb_history, output_path / "ucb_history.csv")

    tx_attempted = int(metrics.get("tx_attempted", 0))
    delivered = int(metrics.get("rx_delivered", metrics.get("delivered", 0)))
    computed_pdr = (delivered / tx_attempted) if tx_attempted > 0 else 0.0

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
            "pdr": float(computed_pdr),
            "throughput_bps": float(metrics.get("throughput_bps", 0.0)),
            "collisions": int(metrics.get("collisions", 0)),
            "tx_attempted": tx_attempted,
            "rx_delivered": delivered,
            "qos_refresh_benchmark": metrics.get("qos_refresh_benchmark", {}),
            "runtime_profile_s": metrics.get("runtime_profile_s", {}),
            "ucb_learning_curve": learning_curve,
        },
        "ucb_config": {
            "source": str(ucb_config_path) if ucb_config_path is not None else "default",
            "lambda_E": ucb_config.lambda_e,
            "exploration_coefficient": ucb_config.exploration_coefficient,
            "reward_window": ucb_config.reward_window,
            "episode_mode": ucb_config.episode.mode,
            "episode_packet_window": ucb_config.episode.packet_window,
            "episode_time_window_s": ucb_config.episode.time_window_s,
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
