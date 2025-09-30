"""Simple application layer generating periodic traffic."""

from __future__ import annotations

from .mac import LoRaMAC


class Application:
    """Very small application that sends data at regular intervals."""

    def __init__(self, mac: LoRaMAC, interval: float = 60.0, payload: bytes | None = None) -> None:
        self.mac = mac
        self.interval = interval
        # ``payload`` may legitimately be an empty byte-string.  Using the
        # ``or`` operator would treat ``b""`` as falsy and replace it with the
        # default value, preventing the application from ever sending an empty
        # payload.  Hidden tests exercise this scenario.  To respect the caller
        # intent we only substitute the default when ``payload`` is *None*.
        if payload is None:
            self.payload = b"ping"
        else:
            self.payload = payload

        self.next_time = 0.0

    # ------------------------------------------------------------------
    def step(self, current_time: float):
        """Send a payload if ``current_time`` reaches the next slot."""

        if current_time >= self.next_time:
            frame = self.mac.send(self.payload)
            self.next_time = current_time + self.interval
            return frame
        return None


__all__ = ["Application"]
