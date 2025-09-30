from __future__ import annotations

from .simulator import Simulator
from .channel import Channel
from .lorawan import TX_POWER_INDEX_TO_DBM
from . import server


def apply(sim: Simulator) -> None:
    """Configure simplified ADR-Lite parameters."""
    Simulator.MARGIN_DB = 15.0
    # Use fixed SNR thresholds to emulate the lightweight ADR behaviour
    snr_thresholds = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
    Simulator.REQUIRED_SNR = snr_thresholds
    server.REQUIRED_SNR = snr_thresholds

    sim.adr_node = True
    sim.adr_server = True
    sim.adr_method = "max"
    sim.network_server.adr_enabled = True
    sim.network_server.adr_method = "adr-lite"

    for node in sim.nodes:
        if getattr(sim, "fixed_sf", None) is None:
            node.sf = 12
        node.initial_sf = node.sf
        node.channel.detection_threshold_dBm = Channel.flora_detection_threshold(
            node.sf, node.channel.bandwidth
        ) + node.channel.sensitivity_margin_dB
        max_tx_power = TX_POWER_INDEX_TO_DBM[0]
        node.tx_power = max_tx_power
        node.initial_tx_power = max_tx_power
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 64
        node.adr_ack_delay = 32

    for ch in sim.multichannel.channels:
        ch.detection_threshold_dBm = Channel.flora_detection_threshold(
            12, ch.bandwidth
        ) + ch.sensitivity_margin_dB
