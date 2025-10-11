"""Tests pour le scénario comparatif QoS vs ADR désactivé."""

from __future__ import annotations

import json
from pathlib import Path

from loraflexsim.scenarios.qos_comparison import run_qos_vs_adr


def test_run_qos_vs_adr_exports_results(tmp_path: Path):
    summary = run_qos_vs_adr(
        num_nodes=4,
        packets_per_node=2,
        packet_interval=30.0,
        duration_s=None,
        seed=1,
        output_dir=tmp_path,
        quiet=True,
    )

    summary_path = summary["summary_path"]
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf8"))
    assert "scenarios" in payload
    assert "gains" in payload

    baseline = payload["scenarios"]["adr_disabled"]
    qos = payload["scenarios"]["qos_enabled"]

    baseline_metrics_path = Path(baseline["metrics_path"])
    qos_metrics_path = Path(qos["metrics_path"])
    baseline_events_path = Path(baseline["events_path"])
    qos_events_path = Path(qos["events_path"])

    assert baseline_metrics_path.exists()
    assert qos_metrics_path.exists()
    assert baseline_events_path.exists()
    assert qos_events_path.exists()

    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf8"))
    qos_metrics = json.loads(qos_metrics_path.read_text(encoding="utf8"))
    baseline_events = json.loads(baseline_events_path.read_text(encoding="utf8"))
    qos_events = json.loads(qos_events_path.read_text(encoding="utf8"))

    assert baseline_metrics["qos_cluster_pdr"] == {}
    assert baseline_metrics["qos_throughput_gini"] == 0.0
    assert qos_metrics["qos_cluster_node_counts"]
    assert qos_metrics["qos_cluster_throughput_bps"]
    assert isinstance(baseline_events, list) and baseline_events
    assert isinstance(qos_events, list) and qos_events
