from dataclasses import dataclass


@dataclass
class LoRaWANFrame:
    """Minimal representation of a LoRaWAN MAC frame."""
    mhdr: int
    fctrl: int
    fcnt: int
    payload: bytes
    confirmed: bool = False


# ---------------------------------------------------------------------------
# LoRaWAN ADR MAC commands (simplified)
# ---------------------------------------------------------------------------

DR_TO_SF = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
SF_TO_DR = {sf: dr for dr, sf in DR_TO_SF.items()}
TX_POWER_INDEX_TO_DBM = {
    0: 20.0,
    1: 17.0,
    2: 14.0,
    3: 11.0,
    4: 8.0,
    5: 5.0,
    6: 2.0,
}
DBM_TO_TX_POWER_INDEX = {int(v): k for k, v in TX_POWER_INDEX_TO_DBM.items()}


@dataclass
class LinkADRReq:
    datarate: int
    tx_power: int
    chmask: int = 0xFFFF
    redundancy: int = 0

    def to_bytes(self) -> bytes:
        dr_tx = ((self.datarate & 0x0F) << 4) | (self.tx_power & 0x0F)
        return bytes([0x03, dr_tx]) + self.chmask.to_bytes(2, "little") + bytes([
            self.redundancy
        ])

    @staticmethod
    def from_bytes(data: bytes) -> "LinkADRReq":
        if len(data) < 5 or data[0] != 0x03:
            raise ValueError("Invalid LinkADRReq")
        dr_tx = data[1]
        datarate = (dr_tx >> 4) & 0x0F
        tx_power = dr_tx & 0x0F
        chmask = int.from_bytes(data[2:4], "little")
        redundancy = data[4]
        return LinkADRReq(datarate, tx_power, chmask, redundancy)


@dataclass
class LinkADRAns:
    status: int = 0b111

    def to_bytes(self) -> bytes:
        return bytes([0x03, self.status])


@dataclass
class LinkCheckReq:
    """LinkCheckReq MAC command"""

    def to_bytes(self) -> bytes:
        return bytes([0x02])


@dataclass
class LinkCheckAns:
    margin: int
    gw_cnt: int

    def to_bytes(self) -> bytes:
        return bytes([0x02, self.margin & 0xFF, self.gw_cnt & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "LinkCheckAns":
        if len(data) < 3 or data[0] != 0x02:
            raise ValueError("Invalid LinkCheckAns")
        return LinkCheckAns(margin=data[1], gw_cnt=data[2])


@dataclass
class DeviceTimeReq:
    """DeviceTimeReq MAC command"""

    def to_bytes(self) -> bytes:
        return bytes([0x0D])


@dataclass
class DeviceTimeAns:
    seconds: int
    fractional: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x0D]) + self.seconds.to_bytes(4, "little") + bytes([self.fractional & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "DeviceTimeAns":
        if len(data) < 6 or data[0] != 0x0D:
            raise ValueError("Invalid DeviceTimeAns")
        secs = int.from_bytes(data[1:5], "little")
        frac = data[5]
        return DeviceTimeAns(secs, frac)


@dataclass
class JoinRequest:
    """Simplified OTAA join request frame."""

    join_eui: int
    dev_eui: int
    dev_nonce: int

    def to_bytes(self) -> bytes:
        return (
            self.join_eui.to_bytes(8, "little")
            + self.dev_eui.to_bytes(8, "little")
            + self.dev_nonce.to_bytes(2, "little")
        )

    @staticmethod
    def from_bytes(data: bytes) -> "JoinRequest":
        if len(data) < 18:
            raise ValueError("Invalid JoinRequest")
        join_eui = int.from_bytes(data[0:8], "little")
        dev_eui = int.from_bytes(data[8:16], "little")
        dev_nonce = int.from_bytes(data[16:18], "little")
        return JoinRequest(join_eui, dev_eui, dev_nonce)


@dataclass
class JoinAccept:
    """Simplified OTAA join accept frame carrying join parameters."""

    app_nonce: int
    net_id: int
    dev_addr: int

    def to_bytes(self) -> bytes:
        return (
            self.app_nonce.to_bytes(3, "little")
            + self.net_id.to_bytes(3, "little")
            + self.dev_addr.to_bytes(4, "little")
        )

    @staticmethod
    def from_bytes(data: bytes) -> "JoinAccept":
        if len(data) < 10:
            raise ValueError("Invalid JoinAccept")
        app_nonce = int.from_bytes(data[0:3], "little")
        net_id = int.from_bytes(data[3:6], "little")
        dev_addr = int.from_bytes(data[6:10], "little")
        return JoinAccept(app_nonce, net_id, dev_addr)


def compute_rx1(end_time: float) -> float:
    """Return the opening time of RX1 window after an uplink."""
    return end_time + 1.0


def compute_rx2(end_time: float) -> float:
    """Return the opening time of RX2 window after an uplink."""
    return end_time + 2.0


def derive_session_keys(app_key: bytes, dev_nonce: int, app_nonce: int, net_id: int) -> tuple[bytes, bytes]:
    """Derive session keys in a simplified manner."""
    import hashlib

    data = (
        app_key
        + dev_nonce.to_bytes(2, "little")
        + app_nonce.to_bytes(3, "little")
        + net_id.to_bytes(3, "little")
    )
    digest = hashlib.sha256(data).digest()
    return digest[:16], digest[16:32]
