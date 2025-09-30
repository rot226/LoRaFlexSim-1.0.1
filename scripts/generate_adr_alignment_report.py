#!/usr/bin/env python3
"""Generate a JSON report comparing ADR decisions against FLoRa."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.compare_flora import replay_flora_txconfig


def build_simulator() -> Simulator:
    """Return a simulator configured like the integration scenario."""

    return Simulator(
        num_nodes=1,
        num_gateways=2,
        area_size=1000.0,
        packet_interval=1200.0,
        first_packet_interval=5.0,
        packets_to_send=0,
        flora_mode=True,
        flora_timing=False,
        adr_server=True,
        adr_method="avg",
        seed=42,
    )


def main() -> None:
    root = ROOT
    data_path = root / "tests" / "integration" / "data" / "flora_multi_gateway_txconfig.json"
    events = json.loads(data_path.read_text(encoding="utf-8"))

    sim = build_simulator()
    report = replay_flora_txconfig(sim, events)

    summary = {
        "scenario": "flora_multi_gateway_txconfig",
        "reference": str(data_path.relative_to(root)),
        "events": len(events),
        "mismatches": len(report["mismatches"]),
        "throttled_events": report["throttled_events"],
        "final_state": report["final_state"],
    }

    output_path = root / "results" / "adr_alignment_report.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
