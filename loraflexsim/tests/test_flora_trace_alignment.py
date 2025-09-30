"""Compare LoRaFlexSim outputs with FLoRa reference traces."""

from __future__ import annotations

import os

import pytest

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.server import ADR_WINDOW_SIZE, MARGIN_DB, REQUIRED_SNR
from loraflexsim.launcher.simulator import Simulator

from .reference_traces import (
    ADR_LOG_REFERENCES,
    ADR_REFERENCES,
    CAPTURE_REFERENCES,
    RSSI_SNR_REFERENCES,
)


def _resolve_tolerance(default: float) -> float:
    """Return the tolerance to use for RSSI/SNR comparisons."""

    override = os.environ.get("FLORA_TRACE_TOLERANCE")
    if override is None:
        return default
    try:
        return float(override)
    except ValueError:
        return default


@pytest.mark.parametrize("trace", RSSI_SNR_REFERENCES)
def test_rssi_snr_matches_flora_reference(trace):
    """Ensure the channel model reproduces FLoRa RSSI/SNR traces."""

    channel = Channel(
        phy_model="flora_full",
        environment="flora",
        shadowing_std=0.0,
        use_flora_curves=True,
        bandwidth=trace.bandwidth_hz,
    )
    rssi, snr = channel.compute_rssi(trace.tx_power_dBm, trace.distance_m, sf=trace.sf)
    tol_rssi = _resolve_tolerance(trace.tol_rssi_dB)
    tol_snr = _resolve_tolerance(trace.tol_snr_dB)
    assert rssi == pytest.approx(trace.expected_rssi_dBm, abs=tol_rssi)
    assert snr == pytest.approx(trace.expected_snr_dB, abs=tol_snr)


@pytest.mark.parametrize("trace", CAPTURE_REFERENCES)
def test_capture_matches_flora_reference(trace):
    """The capture model should agree with FLoRa for simple collisions."""

    channel = Channel(
        phy_model="flora_full",
        environment="flora",
        shadowing_std=0.0,
        flora_capture=True,
        use_flora_curves=True,
    )
    phy = channel.flora_phy
    assert phy is not None
    winners = phy.capture(
        list(trace.rssi_list),
        list(trace.sf_list),
        list(trace.start_list),
        list(trace.end_list),
        list(trace.freq_list),
    )
    assert tuple(bool(x) for x in winners) == trace.expected_winners


@pytest.mark.parametrize("trace", ADR_REFERENCES)
def test_adr_decision_matches_flora_reference(trace):
    """Validate the network server ADR logic against FLoRa expectations."""

    sim = Simulator(num_nodes=1, num_gateways=1, flora_mode=True, mobility=False)
    sim.network_server.adr_enabled = True
    sim.network_server.adr_method = trace.method
    node = sim.nodes[0]
    node.sf = trace.initial_sf
    node.tx_power = trace.initial_power_dBm
    node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    sim.network_server.channel = node.channel

    commands: list[tuple[int, float, int, int]] = []

    def record_command(target_node, payload=b"", confirmed=False, adr_command=None, **kwargs):
        if adr_command:
            commands.append(adr_command)

    sim.network_server.send_downlink = record_command  # type: ignore[assignment]

    noise_floor = node.channel.noise_floor_dBm()
    gateway_id = sim.gateways[0].id
    for event_id, snr in enumerate(trace.snr_values):
        rssi = noise_floor + snr
        sim.network_server.receive(event_id, node.id, gateway_id, rssi=rssi)

    if trace.expected_command is None:
        assert not commands
    else:
        assert commands, "No ADR command emitted"
        last_cmd = commands[-1]
        exp_sf, exp_power, exp_chmask, exp_nbtrans = trace.expected_command
        assert last_cmd[0] == exp_sf
        assert last_cmd[1] == pytest.approx(exp_power, abs=1e-6)
        assert last_cmd[2] == exp_chmask
        assert last_cmd[3] == exp_nbtrans
        assert node.sf == exp_sf
        assert node.tx_power == pytest.approx(exp_power, abs=1e-6)


def test_link_adr_waits_for_twenty_frames_without_adr_ack_req():
    """Ensure LinkADRReq is not emitted before 20 uplinks without ADRACKReq."""

    sim = Simulator(num_nodes=1, num_gateways=1, flora_mode=True, mobility=False)
    server = sim.network_server
    server.adr_enabled = True
    server.adr_method = "max"

    node = sim.nodes[0]
    node.sf = 12
    node.tx_power = 14.0
    node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    server.channel = node.channel

    # Simulate a history of SNR samples as would be available after a previous command.
    gateway_id = sim.gateways[0].id
    node.snr_history = [(gateway_id, 30.0)] * ADR_WINDOW_SIZE
    node.frames_since_last_adr_command = 0

    commands: list[tuple[int, float, int, int]] = []

    def record_command(target_node, payload=b"", confirmed=False, adr_command=None, **kwargs):
        if adr_command:
            commands.append(adr_command)
            target_node.frames_since_last_adr_command = 0

    server.send_downlink = record_command  # type: ignore[assignment]

    noise_floor = node.channel.noise_floor_dBm()
    rssi = noise_floor + 30.0

    window = ADR_WINDOW_SIZE

    for event_id in range(window - 1):
        node.sf = 12
        node.tx_power = 14.0
        node.channel.detection_threshold_dBm = (
            Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
            + node.channel.sensitivity_margin_dB
        )
        server.receive(event_id, node.id, gateway_id, rssi=rssi)

    assert commands == []

    node.sf = 12
    node.tx_power = 14.0
    node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    server.receive(window - 1, node.id, gateway_id, rssi=rssi)
    assert len(commands) == 1

    for event_id in range(window, 2 * window - 1):
        node.sf = 12
        node.tx_power = 14.0
        node.channel.detection_threshold_dBm = (
            Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
            + node.channel.sensitivity_margin_dB
        )
        server.receive(event_id, node.id, gateway_id, rssi=rssi)
        assert len(commands) == 1

    node.sf = 12
    node.tx_power = 14.0
    node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    server.receive(2 * window - 1, node.id, gateway_id, rssi=rssi)
    assert len(commands) == 2


@pytest.mark.parametrize("trace", ADR_LOG_REFERENCES, ids=lambda tr: tr.name)
def test_adr_metric_matches_flora_log(trace):
    """Aggregate SNR and margins match FLoRa-derived logs."""

    sim = Simulator(num_nodes=1, num_gateways=1, flora_mode=True, mobility=False)
    server = sim.network_server
    server.adr_enabled = True
    server.adr_method = trace.method

    node = sim.nodes[0]
    node.sf = trace.initial_sf
    node.tx_power = trace.initial_power_dBm
    node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    server.channel = node.channel

    recorded_metrics: list[tuple[float, float]] = []
    current_sf_for_margin = node.sf

    def record_command(target_node, payload=b"", confirmed=False, adr_command=None, **kwargs):
        nonlocal current_sf_for_margin
        if not adr_command:
            return
        history = tuple(target_node.snr_history)
        assert len(history) == ADR_WINDOW_SIZE
        values = [snr for _, snr in history]
        if trace.method == "avg":
            metric = sum(values) / len(values)
        else:
            metric = max(values)
        required = REQUIRED_SNR.get(current_sf_for_margin, -20.0)
        margin = metric - required - MARGIN_DB
        recorded_metrics.append((metric, margin))
        target_node.frames_since_last_adr_command = 0

    server.send_downlink = record_command  # type: ignore[assignment]

    noise_floor = node.channel.noise_floor_dBm()
    gateway_id = sim.gateways[0].id

    for event_id, snr in enumerate(trace.snr_values):
        rssi = noise_floor + snr
        current_sf_for_margin = node.sf
        server.receive(event_id, node.id, gateway_id, rssi=rssi)

    assert recorded_metrics, "No ADR command recorded"
    metrics, margins = zip(*recorded_metrics)
    assert len(metrics) == len(trace.expected_metrics)
    assert len(margins) == len(trace.expected_margins)
    for got, expected in zip(metrics, trace.expected_metrics):
        assert got == pytest.approx(expected, abs=1e-6)
    for got, expected in zip(margins, trace.expected_margins):
        assert got == pytest.approx(expected, abs=1e-6)
