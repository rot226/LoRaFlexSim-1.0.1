from __future__ import annotations

from .simulator import Simulator


def apply(sim: Simulator) -> None:
    """Configure ADR variant adr_standard_1 (LoRaWAN defaults)."""
    Simulator.MARGIN_DB = 50.0
    sim.adr_node = False
    sim.adr_server = True
    sim.network_server.adr_enabled = True
    for node in sim.nodes:
        node.sf = 12
        node.initial_sf = 12
        node.tx_power = 2.0
        node.initial_tx_power = 2.0
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 64
        node.adr_ack_delay = 32
