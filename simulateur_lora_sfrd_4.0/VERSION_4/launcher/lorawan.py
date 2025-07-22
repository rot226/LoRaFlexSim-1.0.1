from dataclasses import dataclass
from .crypto import aes_encrypt, aes_decrypt, cmac  # noqa: F401


@dataclass
class LoRaWANFrame:
    """Minimal representation of a LoRaWAN MAC frame."""

    mhdr: int
    fctrl: int
    fcnt: int
    payload: bytes
    confirmed: bool = False
    mic: bytes = b""
    encrypted_payload: bytes | None = None


# ---------------------------------------------------------------------------
# LoRaWAN ADR MAC commands (simplified)
# ---------------------------------------------------------------------------

DR_TO_SF = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
SF_TO_DR = {sf: dr for dr, sf in DR_TO_SF.items()}
# Transmission power levels (matching the FLoRa reference values)
TX_POWER_INDEX_TO_DBM = {
    0: 14.0,
    1: 12.0,
    2: 10.0,
    3: 8.0,
    4: 6.0,
    5: 4.0,
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
        return (
            bytes([0x03, dr_tx])
            + self.chmask.to_bytes(2, "little")
            + bytes([self.redundancy])
        )

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

    @staticmethod
    def from_bytes(data: bytes) -> "LinkADRAns":
        if len(data) < 2 or data[0] != 0x03:
            raise ValueError("Invalid LinkADRAns")
        return LinkADRAns(status=data[1])


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
class ResetInd:
    """Inform the network server that the device has reset."""

    minor: int

    def to_bytes(self) -> bytes:
        return bytes([0x01, self.minor & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "ResetInd":
        if len(data) < 2 or data[0] != 0x01:
            raise ValueError("Invalid ResetInd")
        return ResetInd(minor=data[1])


@dataclass
class ResetConf:
    """Acknowledge a ResetInd from the device."""

    minor: int

    def to_bytes(self) -> bytes:
        return bytes([0x01, self.minor & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "ResetConf":
        if len(data) < 2 or data[0] != 0x01:
            raise ValueError("Invalid ResetConf")
        return ResetConf(minor=data[1])


@dataclass
class DutyCycleReq:
    max_duty_cycle: int

    def to_bytes(self) -> bytes:
        return bytes([0x04, self.max_duty_cycle & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "DutyCycleReq":
        if len(data) < 2 or data[0] != 0x04:
            raise ValueError("Invalid DutyCycleReq")
        return DutyCycleReq(max_duty_cycle=data[1])


@dataclass
class RXParamSetupReq:
    rx1_dr_offset: int
    rx2_datarate: int
    frequency: int

    def to_bytes(self) -> bytes:
        dl = ((self.rx1_dr_offset & 0x07) << 4) | (self.rx2_datarate & 0x0F)
        freq = int(self.frequency / 100)
        return bytes([0x05, dl]) + freq.to_bytes(3, "little")

    @staticmethod
    def from_bytes(data: bytes) -> "RXParamSetupReq":
        if len(data) < 5 or data[0] != 0x05:
            raise ValueError("Invalid RXParamSetupReq")
        dl = data[1]
        freq = int.from_bytes(data[2:5], "little") * 100
        return RXParamSetupReq((dl >> 4) & 0x07, dl & 0x0F, freq)


@dataclass
class RXParamSetupAns:
    status: int = 0b111

    def to_bytes(self) -> bytes:
        return bytes([0x05, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "RXParamSetupAns":
        if len(data) < 2 or data[0] != 0x05:
            raise ValueError("Invalid RXParamSetupAns")
        return RXParamSetupAns(status=data[1])


@dataclass
class DevStatusReq:
    def to_bytes(self) -> bytes:
        return bytes([0x06])


@dataclass
class DevStatusAns:
    battery: int
    margin: int

    def to_bytes(self) -> bytes:
        return bytes([0x06, self.battery & 0xFF, self.margin & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "DevStatusAns":
        if len(data) < 3 or data[0] != 0x06:
            raise ValueError("Invalid DevStatusAns")
        return DevStatusAns(battery=data[1], margin=data[2])


@dataclass
class NewChannelReq:
    ch_index: int
    frequency: int
    dr_range: int

    def to_bytes(self) -> bytes:
        freq = int(self.frequency / 100)
        return (
            bytes([0x07, self.ch_index & 0xFF])
            + freq.to_bytes(3, "little")
            + bytes([self.dr_range & 0xFF])
        )

    @staticmethod
    def from_bytes(data: bytes) -> "NewChannelReq":
        if len(data) < 6 or data[0] != 0x07:
            raise ValueError("Invalid NewChannelReq")
        freq = int.from_bytes(data[2:5], "little") * 100
        return NewChannelReq(data[1], freq, data[5])


@dataclass
class NewChannelAns:
    status: int = 0b11

    def to_bytes(self) -> bytes:
        return bytes([0x07, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "NewChannelAns":
        if len(data) < 2 or data[0] != 0x07:
            raise ValueError("Invalid NewChannelAns")
        return NewChannelAns(status=data[1])


@dataclass
class RXTimingSetupReq:
    delay: int

    def to_bytes(self) -> bytes:
        return bytes([0x08, self.delay & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "RXTimingSetupReq":
        if len(data) < 2 or data[0] != 0x08:
            raise ValueError("Invalid RXTimingSetupReq")
        return RXTimingSetupReq(delay=data[1])


@dataclass
class TxParamSetupReq:
    eirp: int
    dwell_time: int

    def to_bytes(self) -> bytes:
        param = ((self.eirp & 0x0F) << 4) | (self.dwell_time & 0x0F)
        return bytes([0x09, param])

    @staticmethod
    def from_bytes(data: bytes) -> "TxParamSetupReq":
        if len(data) < 2 or data[0] != 0x09:
            raise ValueError("Invalid TxParamSetupReq")
        param = data[1]
        return TxParamSetupReq((param >> 4) & 0x0F, param & 0x0F)


@dataclass
class DlChannelReq:
    ch_index: int
    frequency: int

    def to_bytes(self) -> bytes:
        freq = int(self.frequency / 100)
        return bytes([0x0A, self.ch_index & 0xFF]) + freq.to_bytes(3, "little")

    @staticmethod
    def from_bytes(data: bytes) -> "DlChannelReq":
        if len(data) < 5 or data[0] != 0x0A:
            raise ValueError("Invalid DlChannelReq")
        freq = int.from_bytes(data[2:5], "little") * 100
        return DlChannelReq(data[1], freq)


@dataclass
class DlChannelAns:
    status: int = 0b11

    def to_bytes(self) -> bytes:
        return bytes([0x0A, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "DlChannelAns":
        if len(data) < 2 or data[0] != 0x0A:
            raise ValueError("Invalid DlChannelAns")
        return DlChannelAns(status=data[1])


@dataclass
class PingSlotChannelReq:
    frequency: int
    dr: int

    def to_bytes(self) -> bytes:
        freq = int(self.frequency / 100)
        return bytes([0x11]) + freq.to_bytes(3, "little") + bytes([self.dr & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "PingSlotChannelReq":
        if len(data) < 5 or data[0] != 0x11:
            raise ValueError("Invalid PingSlotChannelReq")
        freq = int.from_bytes(data[1:4], "little") * 100
        return PingSlotChannelReq(freq, data[4])


@dataclass
class PingSlotChannelAns:
    status: int = 0b11

    def to_bytes(self) -> bytes:
        return bytes([0x11, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "PingSlotChannelAns":
        if len(data) < 2 or data[0] != 0x11:
            raise ValueError("Invalid PingSlotChannelAns")
        return PingSlotChannelAns(status=data[1])


@dataclass
class PingSlotInfoReq:
    """Request the network server to return the ping slot periodicity."""

    periodicity: int

    def to_bytes(self) -> bytes:
        return bytes([0x10, self.periodicity & 0x07])

    @staticmethod
    def from_bytes(data: bytes) -> "PingSlotInfoReq":
        if len(data) < 2 or data[0] != 0x10:
            raise ValueError("Invalid PingSlotInfoReq")
        return PingSlotInfoReq(data[1] & 0x07)


@dataclass
class PingSlotInfoAns:
    """Acknowledge a PingSlotInfoReq."""

    def to_bytes(self) -> bytes:
        return bytes([0x10])

    @staticmethod
    def from_bytes(data: bytes) -> "PingSlotInfoAns":
        if len(data) < 1 or data[0] != 0x10:
            raise ValueError("Invalid PingSlotInfoAns")
        return PingSlotInfoAns()


@dataclass
class BeaconFreqReq:
    frequency: int

    def to_bytes(self) -> bytes:
        freq = int(self.frequency / 100)
        return bytes([0x13]) + freq.to_bytes(3, "little")

    @staticmethod
    def from_bytes(data: bytes) -> "BeaconFreqReq":
        if len(data) < 4 or data[0] != 0x13:
            raise ValueError("Invalid BeaconFreqReq")
        freq = int.from_bytes(data[1:4], "little") * 100
        return BeaconFreqReq(freq)


@dataclass
class BeaconFreqAns:
    status: int = 0b01

    def to_bytes(self) -> bytes:
        return bytes([0x13, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "BeaconFreqAns":
        if len(data) < 2 or data[0] != 0x13:
            raise ValueError("Invalid BeaconFreqAns")
        return BeaconFreqAns(status=data[1])


@dataclass
class BeaconTimingReq:
    """Request the delay and channel of the next beacon."""

    def to_bytes(self) -> bytes:
        return bytes([0x12])

    @staticmethod
    def from_bytes(data: bytes) -> "BeaconTimingReq":
        if len(data) < 1 or data[0] != 0x12:
            raise ValueError("Invalid BeaconTimingReq")
        return BeaconTimingReq()


@dataclass
class BeaconTimingAns:
    delay: int
    channel: int

    def to_bytes(self) -> bytes:
        return (
            bytes([0x12])
            + self.delay.to_bytes(2, "little")
            + bytes([self.channel & 0xFF])
        )

    @staticmethod
    def from_bytes(data: bytes) -> "BeaconTimingAns":
        if len(data) < 4 or data[0] != 0x12:
            raise ValueError("Invalid BeaconTimingAns")
        delay = int.from_bytes(data[1:3], "little")
        channel = data[3]
        return BeaconTimingAns(delay, channel)


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
        return (
            bytes([0x0D])
            + self.seconds.to_bytes(4, "little")
            + bytes([self.fractional & 0xFF])
        )

    @staticmethod
    def from_bytes(data: bytes) -> "DeviceTimeAns":
        if len(data) < 6 or data[0] != 0x0D:
            raise ValueError("Invalid DeviceTimeAns")
        secs = int.from_bytes(data[1:5], "little")
        frac = data[5]
        return DeviceTimeAns(secs, frac)


@dataclass
class RekeyInd:
    """Start a root key refresh procedure."""

    key_type: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x0B, self.key_type & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "RekeyInd":
        if len(data) < 2 or data[0] != 0x0B:
            raise ValueError("Invalid RekeyInd")
        return RekeyInd(data[1])


@dataclass
class RekeyConf:
    """Acknowledge a RekeyInd from the network."""

    key_type: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x0B, self.key_type & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "RekeyConf":
        if len(data) < 2 or data[0] != 0x0B:
            raise ValueError("Invalid RekeyConf")
        return RekeyConf(data[1])


@dataclass
class ADRParamSetupReq:
    adr_ack_limit: int
    adr_ack_delay: int

    def to_bytes(self) -> bytes:
        param = ((self.adr_ack_limit & 0x0F) << 4) | (self.adr_ack_delay & 0x0F)
        return bytes([0x0C, param])

    @staticmethod
    def from_bytes(data: bytes) -> "ADRParamSetupReq":
        if len(data) < 2 or data[0] != 0x0C:
            raise ValueError("Invalid ADRParamSetupReq")
        param = data[1]
        return ADRParamSetupReq((param >> 4) & 0x0F, param & 0x0F)


@dataclass
class ADRParamSetupAns:
    status: int = 0b111

    def to_bytes(self) -> bytes:
        return bytes([0x0C, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "ADRParamSetupAns":
        if len(data) < 2 or data[0] != 0x0C:
            raise ValueError("Invalid ADRParamSetupAns")
        return ADRParamSetupAns(status=data[1])


@dataclass
class ForceRejoinReq:
    period: int
    rejoin_type: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x0E, self.period & 0xFF, self.rejoin_type & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "ForceRejoinReq":
        if len(data) < 3 or data[0] != 0x0E:
            raise ValueError("Invalid ForceRejoinReq")
        return ForceRejoinReq(period=data[1], rejoin_type=data[2])


@dataclass
class RejoinParamSetupReq:
    max_time_n: int
    max_count_n: int

    def to_bytes(self) -> bytes:
        param = ((self.max_time_n & 0x0F) << 4) | (self.max_count_n & 0x0F)
        return bytes([0x0F, param])

    @staticmethod
    def from_bytes(data: bytes) -> "RejoinParamSetupReq":
        if len(data) < 2 or data[0] != 0x0F:
            raise ValueError("Invalid RejoinParamSetupReq")
        param = data[1]
        return RejoinParamSetupReq((param >> 4) & 0x0F, param & 0x0F)


@dataclass
class RejoinParamSetupAns:
    status: int = 0b11

    def to_bytes(self) -> bytes:
        return bytes([0x0F, self.status])

    @staticmethod
    def from_bytes(data: bytes) -> "RejoinParamSetupAns":
        if len(data) < 2 or data[0] != 0x0F:
            raise ValueError("Invalid RejoinParamSetupAns")
        return RejoinParamSetupAns(status=data[1])


@dataclass
class DeviceModeInd:
    class_mode: str

    def to_bytes(self) -> bytes:
        mapping = {"A": 0, "B": 1, "C": 2}
        return bytes([0x20, mapping.get(self.class_mode, 0) & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "DeviceModeInd":
        if len(data) < 2 or data[0] != 0x20:
            raise ValueError("Invalid DeviceModeInd")
        mapping = {0: "A", 1: "B", 2: "C"}
        return DeviceModeInd(mapping.get(data[1] & 0x03, "A"))


@dataclass
class DeviceModeConf:
    class_mode: str

    def to_bytes(self) -> bytes:
        mapping = {"A": 0, "B": 1, "C": 2}
        return bytes([0x20, mapping.get(self.class_mode, 0) & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "DeviceModeConf":
        if len(data) < 2 or data[0] != 0x20:
            raise ValueError("Invalid DeviceModeConf")
        mapping = {0: "A", 1: "B", 2: "C"}
        return DeviceModeConf(mapping.get(data[1] & 0x03, "A"))


@dataclass
class FragSessionSetupReq:
    index: int
    nb_frag: int
    frag_size: int

    def to_bytes(self) -> bytes:
        return bytes([
            0x21,
            self.index & 0xFF,
            self.nb_frag & 0xFF,
            self.frag_size & 0xFF,
        ])

    @staticmethod
    def from_bytes(data: bytes) -> "FragSessionSetupReq":
        if len(data) < 4 or data[0] != 0x21:
            raise ValueError("Invalid FragSessionSetupReq")
        return FragSessionSetupReq(data[1], data[2], data[3])


@dataclass
class FragSessionSetupAns:
    index: int
    status: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x21, self.index & 0xFF, self.status & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "FragSessionSetupAns":
        if len(data) < 3 or data[0] != 0x21:
            raise ValueError("Invalid FragSessionSetupAns")
        return FragSessionSetupAns(data[1], data[2])


@dataclass
class FragSessionDeleteReq:
    index: int

    def to_bytes(self) -> bytes:
        return bytes([0x22, self.index & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "FragSessionDeleteReq":
        if len(data) < 2 or data[0] != 0x22:
            raise ValueError("Invalid FragSessionDeleteReq")
        return FragSessionDeleteReq(data[1])


@dataclass
class FragSessionDeleteAns:
    status: int = 0

    def to_bytes(self) -> bytes:
        return bytes([0x22, self.status & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "FragSessionDeleteAns":
        if len(data) < 2 or data[0] != 0x22:
            raise ValueError("Invalid FragSessionDeleteAns")
        return FragSessionDeleteAns(data[1])


@dataclass
class FragStatusReq:
    index: int

    def to_bytes(self) -> bytes:
        return bytes([0x23, self.index & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "FragStatusReq":
        if len(data) < 2 or data[0] != 0x23:
            raise ValueError("Invalid FragStatusReq")
        return FragStatusReq(data[1])


@dataclass
class FragStatusAns:
    index: int
    pending: int

    def to_bytes(self) -> bytes:
        return bytes([0x23, self.index & 0xFF, self.pending & 0xFF])

    @staticmethod
    def from_bytes(data: bytes) -> "FragStatusAns":
        if len(data) < 3 or data[0] != 0x23:
            raise ValueError("Invalid FragStatusAns")
        return FragStatusAns(data[1], data[2])


@dataclass
class JoinRequest:
    """Simplified OTAA join request frame."""

    join_eui: int
    dev_eui: int
    dev_nonce: int
    mic: bytes = b""

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
class RejoinRequest:
    """Simplified Rejoin-Request (type 0) frame for re-authentication."""

    rejoin_type: int
    join_eui: int
    dev_eui: int
    rjcount: int
    mic: bytes = b""

    def to_bytes(self) -> bytes:
        return (
            bytes([self.rejoin_type & 0xFF])
            + self.join_eui.to_bytes(8, "little")
            + self.dev_eui.to_bytes(8, "little")
            + self.rjcount.to_bytes(2, "little")
        )

    @staticmethod
    def from_bytes(data: bytes) -> "RejoinRequest":
        if len(data) < 19:
            raise ValueError("Invalid RejoinRequest")
        rtype = data[0]
        join_eui = int.from_bytes(data[1:9], "little")
        dev_eui = int.from_bytes(data[9:17], "little")
        rjcount = int.from_bytes(data[17:19], "little")
        return RejoinRequest(rtype, join_eui, dev_eui, rjcount)


@dataclass
class JoinAccept:
    """Simplified OTAA join accept frame carrying join parameters."""

    app_nonce: int
    net_id: int
    dev_addr: int
    mic: bytes = b""
    encrypted: bytes | None = None

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


def encrypt_join_accept(app_key: bytes, accept: "JoinAccept") -> tuple[bytes, bytes]:
    """Return encrypted join-accept payload and MIC."""
    msg = accept.to_bytes()
    mic = compute_join_mic(app_key, msg)
    padded = msg + mic
    pad_len = (16 - len(padded) % 16) % 16
    encrypted = aes_decrypt(app_key, padded + bytes(pad_len))
    return encrypted, mic


def decrypt_join_accept(
    app_key: bytes, encrypted: bytes, length: int
) -> tuple["JoinAccept", bytes]:
    """Decrypt ``encrypted`` payload and return ``JoinAccept`` and MIC."""
    plain = aes_encrypt(app_key, encrypted)[: length + 4]
    msg = plain[:length]
    mic = plain[length:]
    return JoinAccept.from_bytes(msg), mic


def compute_rx1(end_time: float, rx_delay: float = 1.0) -> float:
    """Return the opening time of RX1 window after an uplink."""
    return end_time + rx_delay


def compute_rx2(end_time: float, rx_delay: float = 1.0) -> float:
    """Return the opening time of RX2 window after an uplink."""
    return end_time + rx_delay + 1.0


def next_beacon_time(
    after_time: float,
    beacon_interval: float,
    *,
    last_beacon: float | None = None,
    drift: float = 0.0,
    loss_limit: float = 2.0,
) -> float:
    """Return the next beacon time after ``after_time``.

    ``drift`` expresses a relative drift of the beacon interval (e.g. ``20e-6``
    for 20 ppm). When ``last_beacon`` is provided, the function keeps applying
    the drift to compute subsequent beacon times. If the elapsed time since the
    last beacon exceeds ``loss_limit`` intervals, the calculation falls back to
    the ideal schedule to simulate a resynchronisation after beacon loss.
    """

    import math

    if last_beacon is None:
        return math.ceil((after_time + 1e-9) / beacon_interval) * beacon_interval

    interval = beacon_interval * (1.0 + drift)
    expected = last_beacon + interval

    if after_time - expected > (loss_limit - 1) * beacon_interval:
        return math.ceil((after_time + 1e-9) / beacon_interval) * beacon_interval

    if expected <= after_time:
        steps = math.ceil((after_time - expected) / interval)
        expected += steps * interval
    return expected


def next_ping_slot_time(
    last_beacon_time: float,
    after_time: float,
    periodicity: int,
    ping_slot_interval: float,
    ping_slot_offset: float,
    *,
    beacon_drift: float = 0.0,
) -> float:
    """Return the next ping slot time after ``after_time``.

    ``beacon_drift`` can be used to apply the same drift as returned by
    :func:`next_beacon_time` when computing the window relative to the last
    beacon.
    """
    import math

    first_slot = last_beacon_time + beacon_drift + ping_slot_offset
    if after_time <= first_slot:
        return first_slot
    interval = ping_slot_interval * (2**periodicity)
    slots = math.ceil((after_time - first_slot) / interval)
    return first_slot + slots * interval


# ---------------------------------------------------------------------------
# LoRaWAN security helpers (AES encryption and MIC)
# ---------------------------------------------------------------------------


def encrypt_payload(
    app_skey: bytes,
    devaddr: int,
    fcnt: int,
    direction: int,
    payload: bytes,
) -> bytes:
    """Encrypt ``payload`` using the LoRaWAN payload encryption scheme."""
    blocks = []
    for i in range(1, (len(payload) + 15) // 16 + 1):
        a = (
            bytes([0x01, 0x00, 0x00, 0x00, 0x00])
            + bytes([direction & 0x01])
            + devaddr.to_bytes(4, "little")
            + fcnt.to_bytes(4, "little")
            + bytes([0x00, i])
        )
        s = aes_encrypt(app_skey, a)
        start = (i - 1) * 16
        block = payload[start : start + 16]
        blocks.append(bytes(x ^ y for x, y in zip(block, s)))
    return b"".join(blocks)[: len(payload)]


def compute_mic(
    nwk_skey: bytes,
    devaddr: int,
    fcnt: int,
    direction: int,
    msg: bytes,
) -> bytes:
    """Compute MIC of ``msg`` using ``nwk_skey``."""
    b0 = (
        bytes([0x49, 0x00, 0x00, 0x00, 0x00])
        + bytes([direction & 0x01])
        + devaddr.to_bytes(4, "little")
        + fcnt.to_bytes(4, "little")
        + bytes([0x00, len(msg)])
    )
    mac = cmac(nwk_skey, b0 + msg)
    return mac[:4]


def encrypt_multicast_payload(
    mc_app_skey: bytes, mc_addr: int, fcnt: int, payload: bytes
) -> bytes:
    """Encrypt multicast ``payload`` (downlink direction)."""
    return encrypt_payload(mc_app_skey, mc_addr, fcnt, 1, payload)


def compute_multicast_mic(
    mc_nwk_skey: bytes, mc_addr: int, fcnt: int, msg: bytes
) -> bytes:
    """Compute MIC for multicast messages (downlink direction)."""
    return compute_mic(mc_nwk_skey, mc_addr, fcnt, 1, msg)


def compute_join_mic(app_key: bytes, msg: bytes) -> bytes:
    """Compute MIC for join request/accept messages."""
    return cmac(app_key, msg)[:4]


def compute_rejoin_mic(app_key: bytes, msg: bytes) -> bytes:
    """Compute MIC for rejoin-request messages (type 0 only)."""
    return cmac(app_key, msg)[:4]


def derive_session_keys(
    app_key: bytes, dev_nonce: int, app_nonce: int, net_id: int
) -> tuple[bytes, bytes]:
    """Derive NwkSKey and AppSKey following the LoRaWAN 1.0.x specification."""

    def _derive(k: int) -> bytes:
        return aes_encrypt(
            app_key,
            bytes([k])
            + app_nonce.to_bytes(3, "little")
            + net_id.to_bytes(3, "little")
            + dev_nonce.to_bytes(2, "little")
            + bytes(7),
        )

    nwk_skey = _derive(0x01)
    app_skey = _derive(0x02)
    return nwk_skey, app_skey


def validate_frame(
    frame: LoRaWANFrame,
    nwk_skey: bytes,
    app_skey: bytes,
    devaddr: int,
    direction: int,
) -> bool:
    """Check MIC and decrypt payload in ``frame``."""
    if frame.encrypted_payload is not None:
        if (
            compute_mic(
                nwk_skey, devaddr, frame.fcnt, direction, frame.encrypted_payload
            )
            != frame.mic
        ):
            return False
        frame.payload = encrypt_payload(
            app_skey, devaddr, frame.fcnt, direction, frame.encrypted_payload
        )
        return True
    return (
        compute_mic(nwk_skey, devaddr, frame.fcnt, direction, frame.payload)
        == frame.mic
    )


def validate_join_request(req: JoinRequest, app_key: bytes) -> bool:
    """Return ``True`` if ``req`` has a valid MIC."""
    return compute_join_mic(app_key, req.to_bytes()) == req.mic


def validate_rejoin_request(req: RejoinRequest, app_key: bytes) -> bool:
    """Return ``True`` if ``req`` has a valid MIC (type 0)."""
    return compute_rejoin_mic(app_key, req.to_bytes()) == req.mic
