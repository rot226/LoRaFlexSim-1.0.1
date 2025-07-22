from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from .lorawan import JoinAccept, JoinRequest, RejoinRequest


@dataclass
class JoinServer:
    """Minimal OTAA join server handling MIC validation and key derivation."""

    net_id: int = 0
    devices: dict[tuple[int, int], bytes] = field(default_factory=dict)
    last_devnonce: dict[tuple[int, int], int] = field(default_factory=dict)
    last_rjcount0: dict[tuple[int, int], int] = field(default_factory=dict)
    session_keys: dict[tuple[int, int], tuple[bytes, bytes]] = field(
        default_factory=dict
    )
    next_devaddr: int = 1
    app_nonce: int = 1

    def register(self, join_eui: int, dev_eui: int, app_key: bytes) -> None:
        """Register a device and its AppKey."""
        if len(app_key) != 16:
            raise ValueError("Invalid AppKey")
        self.devices[(join_eui, dev_eui)] = app_key

    def get_session_keys(
        self, join_eui: int, dev_eui: int
    ) -> tuple[bytes, bytes] | None:
        """Return stored (NwkSKey, AppSKey) for a device if known."""
        return self.session_keys.get((join_eui, dev_eui))

    def handle_join(self, req: "JoinRequest") -> tuple["JoinAccept", bytes, bytes]:
        """Validate ``req`` and return a join-accept frame with session keys."""
        from .lorawan import (
            compute_join_mic,
            derive_session_keys,
            encrypt_join_accept,
            JoinAccept,
        )

        key = (req.join_eui, req.dev_eui)
        app_key = self.devices.get(key)
        if app_key is None:
            raise KeyError("Unknown device")

        if compute_join_mic(app_key, req.to_bytes()) != req.mic:
            raise ValueError("Invalid MIC")

        last = self.last_devnonce.get(key)
        if last is not None and req.dev_nonce <= last:
            raise ValueError("DevNonce reused")
        self.last_devnonce[key] = req.dev_nonce

        app_nonce = self.app_nonce & 0xFFFFFF
        self.app_nonce += 1
        dev_addr = self.next_devaddr
        self.next_devaddr += 1

        nwk_skey, app_skey = derive_session_keys(
            app_key, req.dev_nonce, app_nonce, self.net_id
        )

        accept = JoinAccept(app_nonce, self.net_id, dev_addr)
        enc, mic = encrypt_join_accept(app_key, accept)
        accept.encrypted = enc
        accept.mic = mic
        self.session_keys[key] = (nwk_skey, app_skey)
        return accept, nwk_skey, app_skey

    def handle_rejoin(self, req: "RejoinRequest") -> tuple["JoinAccept", bytes, bytes]:
        """Process a RejoinRequest type 0 and return a join-accept."""
        from .lorawan import (
            compute_rejoin_mic,
            derive_session_keys,
            encrypt_join_accept,
            JoinAccept,
        )

        if req.rejoin_type != 0:
            raise NotImplementedError("Only RejoinRequest type 0 supported")

        key = (req.join_eui, req.dev_eui)
        app_key = self.devices.get(key)
        if app_key is None:
            raise KeyError("Unknown device")

        if compute_rejoin_mic(app_key, req.to_bytes()) != req.mic:
            raise ValueError("Invalid MIC")

        last = self.last_rjcount0.get(key)
        if last is not None and req.rjcount <= last:
            raise ValueError("RJcount reused")
        self.last_rjcount0[key] = req.rjcount

        app_nonce = self.app_nonce & 0xFFFFFF
        self.app_nonce += 1
        dev_addr = self.next_devaddr
        self.next_devaddr += 1

        nwk_skey, app_skey = derive_session_keys(
            app_key, req.rjcount, app_nonce, self.net_id
        )

        accept = JoinAccept(app_nonce, self.net_id, dev_addr)
        enc, mic = encrypt_join_accept(app_key, accept)
        accept.encrypted = enc
        accept.mic = mic
        self.session_keys[key] = (nwk_skey, app_skey)
        return accept, nwk_skey, app_skey
