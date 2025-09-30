from __future__ import annotations

from .simulator import Simulator
from .channel import Channel


def apply(sim: Simulator) -> None:
    """Configure ADR variant adr_2."""
    Simulator.MARGIN_DB = 30.0
    sim.adr_node = True
    sim.adr_server = True
    sim.network_server.adr_enabled = True
    for node in sim.nodes:
        node.sf = 7
        node.initial_sf = 7
        node.channel.detection_threshold_dBm = Channel.flora_detection_threshold(
            node.sf, node.channel.bandwidth
        ) + node.channel.sensitivity_margin_dB
        node.tx_power = 14.0
        node.initial_tx_power = 14.0
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 8
        node.adr_ack_delay = 4
    for ch in sim.multichannel.channels:
        ch.detection_threshold_dBm = Channel.flora_detection_threshold(7, ch.bandwidth) + ch.sensitivity_margin_dB
