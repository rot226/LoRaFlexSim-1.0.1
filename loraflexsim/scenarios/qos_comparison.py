"""Scénarios comparatifs QoS vs ADR désactivé."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.qos import QoSManager

__all__ = ["run_qos_vs_adr"]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "qos_comparison"


@dataclass
class ScenarioResult:
    """Résultat d'un scénario simulé."""

    label: str
    metrics_path: Path
    events_path: Path
    metrics: Mapping[str, Any]
    events_count: int


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(payload)
    with path.open("w", encoding="utf8") as handle:
        json.dump(sanitized, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _configure_qos(sim: Simulator) -> None:
    manager = QoSManager()
    manager.configure_clusters(
        2,
        proportions=[0.6, 0.4],
        arrival_rates=[1.0 / 180.0, 1.0 / 90.0],
        pdr_targets=[0.95, 0.9],
    )
    manager.apply(sim, "MixRA-Opt")


def _run_single(
    label: str,
    *,
    enable_qos: bool,
    num_nodes: int,
    packets_per_node: int,
    packet_interval: float,
    duration_s: float | None,
    seed: int,
    output_dir: Path,
    quiet: bool,
) -> ScenarioResult:
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
        duty_cycle=None,
        mobility=False,
        seed=seed,
        payload_size_bytes=20,
    )
    if enable_qos:
        _configure_qos(sim)
    else:
        setattr(sim, "qos_clusters_config", {})
        setattr(sim, "qos_node_clusters", {})
    if duration_s is None:
        sim.run()
    else:
        sim.run(max_time=duration_s)
    metrics = sim.get_metrics()
    events = list(sim.events_log)
    metrics_path = output_dir / f"{label}_metrics.json"
    events_path = output_dir / f"{label}_events.json"
    _write_json(metrics_path, metrics)
    _write_json(events_path, events)
    if not quiet:
        pdr = float(metrics.get("PDR", 0.0)) * 100.0
        throughput = float(metrics.get("throughput_bps", 0.0))
        print(f"{label}: PDR={pdr:.2f}% throughput={throughput:.2f} bps")
    return ScenarioResult(
        label=label,
        metrics_path=metrics_path,
        events_path=events_path,
        metrics=metrics,
        events_count=len(events),
    )


def run_qos_vs_adr(
    *,
    num_nodes: int = 24,
    packets_per_node: int = 6,
    packet_interval: float = 180.0,
    duration_s: float | None = None,
    seed: int = 1,
    output_dir: Path | None = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Exécute deux scénarios (QoS actif vs ADR désactivé) et exporte les métriques.

    :returns: dictionnaire comprenant les chemins des exports et un résumé des gains.
    """

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adr_disabled = _run_single(
        "adr_disabled",
        enable_qos=False,
        num_nodes=num_nodes,
        packets_per_node=packets_per_node,
        packet_interval=packet_interval,
        duration_s=duration_s,
        seed=seed,
        output_dir=output_dir,
        quiet=quiet,
    )
    qos_enabled = _run_single(
        "qos_enabled",
        enable_qos=True,
        num_nodes=num_nodes,
        packets_per_node=packets_per_node,
        packet_interval=packet_interval,
        duration_s=duration_s,
        seed=seed,
        output_dir=output_dir,
        quiet=quiet,
    )

    baseline_metrics = adr_disabled.metrics
    qos_metrics = qos_enabled.metrics
    pdr_gain = float(qos_metrics.get("PDR", 0.0)) - float(baseline_metrics.get("PDR", 0.0))
    throughput_gain = float(qos_metrics.get("throughput_bps", 0.0)) - float(
        baseline_metrics.get("throughput_bps", 0.0)
    )

    summary_payload = {
        "settings": {
            "num_nodes": num_nodes,
            "packets_per_node": packets_per_node,
            "packet_interval": packet_interval,
            "duration_s": duration_s,
            "seed": seed,
        },
        "scenarios": {
            adr_disabled.label: {
                "metrics_path": str(adr_disabled.metrics_path),
                "events_path": str(adr_disabled.events_path),
                "PDR": float(baseline_metrics.get("PDR", 0.0)),
                "throughput_bps": float(baseline_metrics.get("throughput_bps", 0.0)),
                "events": adr_disabled.events_count,
                "qos_throughput_gini": float(
                    baseline_metrics.get("qos_throughput_gini", 0.0)
                ),
            },
            qos_enabled.label: {
                "metrics_path": str(qos_enabled.metrics_path),
                "events_path": str(qos_enabled.events_path),
                "PDR": float(qos_metrics.get("PDR", 0.0)),
                "throughput_bps": float(qos_metrics.get("throughput_bps", 0.0)),
                "events": qos_enabled.events_count,
                "qos_throughput_gini": float(
                    qos_metrics.get("qos_throughput_gini", 0.0)
                ),
            },
        },
        "gains": {
            "PDR": pdr_gain,
            "throughput_bps": throughput_gain,
        },
    }

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "summary": summary_payload,
    }
