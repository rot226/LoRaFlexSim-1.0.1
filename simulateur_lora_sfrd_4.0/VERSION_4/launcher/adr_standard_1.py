from __future__ import annotations

from .simulator import Simulator
from . import server
from .advanced_channel import AdvancedChannel


def apply(sim: Simulator, *, degrade_channel: bool = True) -> None:
    """Configure ADR variant ``adr_standard_1`` (LoRaWAN defaults).

    Parameters
    ----------
    sim : Simulator
        Instance to modify in-place.
    degrade_channel : bool, optional
        When ``True`` (default) apply additional propagation
        impairments to reduce the Packet Delivery Ratio.  Only the ADR1
        preset applies these changes.
    """
    Simulator.MARGIN_DB = 15.0
    server.MARGIN_DB = Simulator.MARGIN_DB
    sim.adr_node = False
    sim.adr_server = True
    sim.network_server.adr_enabled = True
    for node in sim.nodes:
        node.sf = 12
        node.initial_sf = 12
        node.tx_power = 14.0
        node.initial_tx_power = 14.0
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 64
        node.adr_ack_delay = 32

    if degrade_channel:
        for ch in sim.multichannel.channels:
            base = ch.base if isinstance(ch, AdvancedChannel) else ch
            # Slightly stronger constant interference
            base.interference_dB = max(base.interference_dB, 8.0)
            # Increase the fast fading variance a little more
            base.fast_fading_std = max(base.fast_fading_std, 4.5)
            # Raise the extra path loss exponent just a bit further
            base.path_loss_exp = max(base.path_loss_exp, 3.6)
            # Detection threshold marginally above previous value
            base.detection_threshold_dBm = max(base.detection_threshold_dBm, -92.0)
            # Allow slow noise variations with slightly higher amplitude
            base.noise_floor_std = max(base.noise_floor_std, 2.1)
            if isinstance(ch, AdvancedChannel):
                ch.fading = "rayleigh"
                ch.weather_loss_dB_per_km = max(ch.weather_loss_dB_per_km, 1.2)
        sim.detection_threshold_dBm = -92.0
