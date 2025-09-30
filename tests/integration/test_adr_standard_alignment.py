"""Compare ADR Standard decisions against a reference FLoRa trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.compare_flora import replay_flora_txconfig

DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "flora_multi_gateway_txconfig.json"
)


@pytest.mark.integration
def test_adr_standard_alignment_with_flora_trace():
    """Ensure ADR standard reproduces the FLoRa TXCONFIG decisions."""

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        events = json.load(handle)

    sim = Simulator(
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
    node = sim.nodes[0]
    initial_sf = node.sf
    initial_power = node.tx_power

    report = replay_flora_txconfig(sim, events)

    assert not report["mismatches"]

    last_decision = None
    for result, entry in zip(report["results"], events, strict=True):
        best_gateway = entry["best_gateway"]
        expected_snr = entry["gateways"][str(best_gateway)]["snr"]
        assert result["best_gateway"] == best_gateway
        assert result["recorded_snr"] == expected_snr

        expected_command = entry["expected_command"]
        if expected_command:
            if result["throttled"]:
                assert result["decision"] is None
            else:
                assert result["decision"] is not None
                assert result["decision"]["sf"] == expected_command["sf"]
                assert result["decision"]["tx_power"] == expected_command["tx_power"]
                assert result["decision"]["gateway_id"] == expected_command["gateway_id"]
                last_decision = result["decision"]
        else:
            assert result["decision"] is None

    if last_decision:
        assert report["final_state"]["sf"] == last_decision["sf"]
        assert report["final_state"]["tx_power"] == last_decision["tx_power"]
    else:
        assert report["final_state"]["sf"] == initial_sf
        assert report["final_state"]["tx_power"] == initial_power
