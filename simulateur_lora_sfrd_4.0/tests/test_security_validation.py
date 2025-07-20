import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.lorawan import (  # noqa: E402
    LoRaWANFrame,
    encrypt_payload,
    compute_mic,
    validate_frame,
    validate_join_request,
    compute_join_mic,
    JoinRequest,
)


def test_validate_frame_and_join_request():
    nwk = bytes(range(16))
    app = bytes(range(16, 32))
    devaddr = 0x01020304
    payload = b"hello"

    enc = encrypt_payload(app, devaddr, 1, 0, payload)
    mic = compute_mic(nwk, devaddr, 1, 0, enc)
    frame = LoRaWANFrame(
        mhdr=0x40,
        fctrl=0,
        fcnt=1,
        payload=b"",
        confirmed=False,
        mic=mic,
        encrypted_payload=enc,
    )
    assert validate_frame(frame, nwk, app, devaddr, 0)
    assert frame.payload == payload

    bad = LoRaWANFrame(
        mhdr=0x40,
        fctrl=0,
        fcnt=1,
        payload=b"",
        confirmed=False,
        mic=b"\x00\x00\x00\x00",
        encrypted_payload=enc,
    )
    assert not validate_frame(bad, nwk, app, devaddr, 0)

    key = bytes(range(16))
    req = JoinRequest(1, 2, 1)
    req.mic = compute_join_mic(key, req.to_bytes())
    assert validate_join_request(req, key)

    req.mic = b"\x00\x00\x00\x00"
    assert not validate_join_request(req, key)
