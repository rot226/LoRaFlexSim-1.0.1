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


def compute_rx1(end_time: float) -> float:
    """Return the opening time of RX1 window after an uplink."""
    return end_time + 1.0


def compute_rx2(end_time: float) -> float:
    """Return the opening time of RX2 window after an uplink."""
    return end_time + 2.0
