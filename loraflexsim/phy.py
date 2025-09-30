"""Very small helper representing the LoRa physical layer."""

from __future__ import annotations

import inspect
import random
import numpy as np

from .loranode import Node


class LoRaPHY:
    """Interface to :class:`~launcher.channel.Channel` for a given node."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.channel = node.channel

    # ------------------------------------------------------------------
    def airtime(self, payload_size: int) -> float:
        """Return packet airtime for the current data rate."""

        return self.channel.airtime(self.node.sf, payload_size)

    # ------------------------------------------------------------------
    def transmit(
        self,
        dest: Node,
        payload_size: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float, float, bool]:
        """Simulate a transmission to ``dest``.

        Parameters
        ----------
        dest : Node
            Receiving node or gateway.
        payload_size : int
            Size of the payload in bytes.

        Returns
        -------
        tuple
            ``(rssi, snr, airtime, success)`` for the link.
        """

        distance = self.node.distance_to(dest)
        rssi, snr = self.channel.compute_rssi(
            self.node.tx_power,
            distance,
            self.node.sf,
            freq_offset_hz=self.node.current_freq_offset,
            sync_offset_s=self.node.current_sync_offset,
        )
        channel = self.channel
        flora_phy = getattr(channel, "flora_phy", None)
        per = None
        if flora_phy is not None and (
            getattr(channel, "use_flora_curves", False)
            or getattr(channel, "phy_model", "").startswith("flora")
        ):
            per_kwargs = {}
            try:
                params = inspect.signature(flora_phy.packet_error_rate).parameters
            except (TypeError, ValueError):
                params = {}
            if "per_model" in params:
                per_model = getattr(channel, "flora_per_model", "logistic")
                if per_model is not None:
                    per_kwargs["per_model"] = per_model
            per = flora_phy.packet_error_rate(
                snr,
                self.node.sf,
                payload_bytes=payload_size,
                **per_kwargs,
            )
        if per is None and hasattr(channel, "packet_error_rate"):
            per = channel.packet_error_rate(
                snr, self.node.sf, payload_bytes=payload_size
            )
        if per is None:
            per = 0.0
        rand = rng.random() if rng is not None else random.random()
        success = per < 1.0 and rand >= per
        return rssi, snr, self.airtime(payload_size), success


__all__ = ["LoRaPHY"]
