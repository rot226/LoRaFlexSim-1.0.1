from __future__ import annotations

from .simulator import Simulator
from .channel import Channel


def apply(sim: Simulator) -> None:
    """Configure the ADR-Max algorithm.

    ADR-Max selects the highest spreading factor that still supports the link.
    Nodes start with a conservative configuration so that the network server
    can later optimise the spreading factor based on the measured SNR of the
    first uplinks.
    """
    # Installation margin used both by the simulator and the network server
    Simulator.MARGIN_DB = 15.0
    from . import server as ns

    ns.MARGIN_DB = 15.0

    sim.adr_node = True
    sim.adr_server = True
    sim.network_server.adr_enabled = True
    sim.network_server.adr_method = "adr-max"

    for node in sim.nodes:
        if getattr(sim, "fixed_sf", None) is None:
            node.sf = 12
        node.initial_sf = node.sf
        node.channel.detection_threshold_dBm = (
            Channel.flora_detection_threshold(node.sf, node.channel.bandwidth)
            + node.channel.sensitivity_margin_dB
        )
        node.tx_power = 14.0
        node.initial_tx_power = 14.0
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 64
        node.adr_ack_delay = 32

    for ch in sim.multichannel.channels:
        ch.detection_threshold_dBm = (
            Channel.flora_detection_threshold(12, ch.bandwidth)
            + ch.sensitivity_margin_dB
        )
