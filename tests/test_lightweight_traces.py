import gzip
import json

from scripts.simulation_analysis_utils import (
    cache_metrics_ready,
    export_lightweight_trace,
)


class DummyNode:
    def __init__(self, identifier: int, energy: float, delivered: int) -> None:
        self.id = identifier
        self.tx_attempted = delivered
        self.rx_delivered = delivered
        self.energy_consumed = energy


def test_export_lightweight_trace_csv(tmp_path):
    records = [
        {"time": 0.0, "node_id": 1, "result": "Success"},
        {"time": 1.0, "node_id": 2, "result": "Collision"},
    ]
    out = tmp_path / "trace.csv.gz"
    export_lightweight_trace(records, out, ["time", "node_id"])
    with gzip.open(out, "rt") as handle:
        content = handle.read().strip().splitlines()
    assert content[0] == "time,node_id"
    assert content[1] == "0.0,1"


def test_cache_metrics_ready(tmp_path):
    events = [
        {"result": "Success", "start_time": 0.0, "end_time": 1.0},
        {"result": "Success", "start_time": 1.0, "end_time": 2.5},
    ]
    nodes = [DummyNode(1, 0.5, 2), DummyNode(2, 0.7, 1)]
    out = tmp_path / "metrics.json"
    cache_metrics_ready(events, nodes, out)
    data = json.loads(out.read_text())
    assert data["attempts"] == 3
    assert data["deliveries"] == 3
    assert data["node_count"] == 2
