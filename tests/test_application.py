import types

from loraflexsim.application import Application


class DummyMAC:
    """Minimal MAC layer used to inspect outgoing payloads."""

    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)
        return payload


def test_preserves_empty_payload():
    """Application should not replace an explicitly empty payload."""
    mac = DummyMAC()
    app = Application(mac, payload=b"")

    # ``payload`` must be stored verbatim
    assert app.payload == b""

    # First step at t=0 should trigger a transmission of that empty payload
    frame = app.step(0.0)
    assert frame == b""
    assert mac.sent == [b""]
