"""Simplified LoRa MAC layer implementation."""

from __future__ import annotations

from .loranode import Node


class LoRaMAC:
    """Basic MAC layer tied to a :class:`Node` instance.

    This helper exposes minimal methods used by the examples and tests
    to build uplink frames and process downlink commands.  It delegates
    most of the work to :class:`~loraflexsim.launcher.node.Node`.
    """

    def __init__(self, node: Node) -> None:
        self.node = node

    # ------------------------------------------------------------------
    def send(self, payload: bytes, confirmed: bool = False):
        """Prepare an uplink frame for ``payload``.

        Parameters
        ----------
        payload : bytes
            Raw application payload.
        confirmed : bool, optional
            Request confirmation from the server.

        Returns
        -------
        object
            The ``LoRaWANFrame`` created by :meth:`Node.prepare_uplink`.
        """

        return self.node.prepare_uplink(payload, confirmed)

    # ------------------------------------------------------------------
    def process_downlink(self, frame) -> None:
        """Forward a downlink frame to the underlying node."""

        self.node.handle_downlink(frame)


__all__ = ["LoRaMAC"]
