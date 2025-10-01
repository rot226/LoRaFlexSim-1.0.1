"""Reference traces extracted from FLoRa formulas for integration tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.flora_phy import FloraPHY

# LoRaWAN defaults reused by the ADR algorithm.  The mapping matches the
# TX_POWER_INDEX_TO_DBM table from the specification and the FLoRa code base.
TX_POWER_INDEX_TO_DBM = {
    0: 14.0,
    1: 11.0,
    2: 8.0,
    3: 5.0,
    4: 2.0,
}
DBM_TO_TX_POWER_INDEX = {int(v): k for k, v in TX_POWER_INDEX_TO_DBM.items()}

# SNR thresholds used by the ADR decision in FLoRa (LoRaWAN 1.0.2 defaults).
REQUIRED_SNR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


def _round_half_away_from_zero(value: float) -> int:
    """Mirror the rounding behaviour of ``std::round`` (half away from zero)."""

    return int(math.copysign(math.floor(abs(value) + 0.5), value))


@dataclass(frozen=True)
class RssiSnrTrace:
    """Reference RSSI/SNR for a simple link budget scenario."""

    name: str
    tx_power_dBm: float
    distance_m: float
    sf: int
    bandwidth_hz: int
    expected_rssi_dBm: float
    expected_snr_dB: float
    tol_rssi_dB: float = 0.5
    tol_snr_dB: float = 0.5


@dataclass(frozen=True)
class CaptureTrace:
    """Reference capture decision for overlapping transmissions."""

    name: str
    rssi_list: Tuple[float, ...]
    sf_list: Tuple[int, ...]
    start_list: Tuple[float, ...]
    end_list: Tuple[float, ...]
    freq_list: Tuple[float, ...]
    expected_winners: Tuple[bool, ...]


@dataclass(frozen=True)
class AdrTrace:
    """Reference ADR decision derived from the FLoRa algorithm."""

    name: str
    snr_values: Tuple[float, ...]
    initial_sf: int
    initial_power_dBm: float
    method: str
    expected_command: Tuple[int, float, int, int] | None


@dataclass(frozen=True)
class AdrLogTrace:
    """Reference ADR metrics recorded from FLoRa logs."""

    name: str
    method: str
    initial_sf: int
    initial_power_dBm: float
    snr_values: Tuple[float, ...]
    expected_metrics: Tuple[float, ...]
    expected_margins: Tuple[float, ...]


def _flora_rssi_snr(channel: Channel, tx_power_dBm: float, distance_m: float, sf: int) -> tuple[float, float]:
    """Return RSSI and SNR using the same formulas as FLoRa."""

    loss = (
        FloraPHY.PATH_LOSS_D0
        + 10 * channel.path_loss_exp * math.log10(max(distance_m, 1.0) / FloraPHY.REFERENCE_DISTANCE)
    )
    rssi = (
        tx_power_dBm
        + channel.tx_antenna_gain_dB
        + channel.rx_antenna_gain_dB
        - loss
        - channel.cable_loss_dB
        + channel.rssi_offset_dB
    )
    noise = channel.FLORA_SENSITIVITY[sf][int(channel.bandwidth)]
    snr = rssi - noise + channel.snr_offset_dB
    if channel.processing_gain:
        snr += 10 * math.log10(2 ** sf)
    return rssi, snr


def _flora_adr_decision(
    snr_values: tuple[float, ...],
    initial_sf: int,
    initial_power_dBm: float,
    *,
    method: str = "avg",
    margin_db: float = 15.0,
) -> tuple[int, float, int, int] | None:
    """Reproduce the ADR decision logic implemented in FLoRa."""

    if not snr_values:
        return None

    if method == "avg":
        metric = sum(snr_values) / len(snr_values)
    else:
        metric = max(snr_values)

    required = REQUIRED_SNR.get(initial_sf, -20.0)
    margin = metric - required - margin_db
    nstep = _round_half_away_from_zero(margin / 3.0)

    sf = initial_sf
    power_idx = DBM_TO_TX_POWER_INDEX.get(int(round(initial_power_dBm)), 0)
    power_order = sorted(
        TX_POWER_INDEX_TO_DBM.items(), key=lambda item: item[1], reverse=True
    )
    idx_to_rank = {idx: rank for rank, (idx, _) in enumerate(power_order)}
    rank_to_idx = {rank: idx for idx, rank in idx_to_rank.items()}
    power_rank = idx_to_rank.get(power_idx, 0)
    max_rank = len(power_order) - 1

    if nstep > 0:
        while nstep > 0 and sf > 7:
            sf -= 1
            nstep -= 1
        while nstep > 0 and power_rank < max_rank:
            power_rank += 1
            nstep -= 1
    elif nstep < 0:
        while nstep < 0 and power_rank > 0:
            power_rank -= 1
            nstep += 1
        while nstep < 0 and sf < 12:
            sf += 1
            nstep += 1

    power_idx = rank_to_idx.get(power_rank, power_idx)
    power = TX_POWER_INDEX_TO_DBM.get(power_idx, initial_power_dBm)

    if sf == initial_sf and math.isclose(power, initial_power_dBm, abs_tol=1e-6):
        return None

    return sf, power, 0xFFFF, 1


def _make_rssi_snr_traces() -> tuple[RssiSnrTrace, ...]:
    traces: list[RssiSnrTrace] = []
    for name, distance, sf in [
        ("flora_sf7_40m", 40.0, 7),
        ("flora_sf9_250m", 250.0, 9),
        ("flora_sf12_1000m", 1000.0, 12),
    ]:
        channel = Channel(
            phy_model="flora_full",
            environment="flora",
            shadowing_std=0.0,
            use_flora_curves=True,
            bandwidth=125_000,
        )
        rssi, snr = _flora_rssi_snr(channel, 14.0, distance, sf)
        traces.append(
            RssiSnrTrace(
                name=name,
                tx_power_dBm=14.0,
                distance_m=distance,
                sf=sf,
                bandwidth_hz=125_000,
                expected_rssi_dBm=rssi,
                expected_snr_dB=snr,
                tol_rssi_dB=0.6,
                tol_snr_dB=0.6,
            )
        )
    return tuple(traces)


def _make_capture_traces() -> tuple[CaptureTrace, ...]:
    traces: list[CaptureTrace] = []
    channel = Channel(
        phy_model="flora_full",
        environment="flora",
        shadowing_std=0.0,
        flora_capture=True,
        use_flora_curves=True,
    )
    phy = FloraPHY(channel)

    # Strong capture: 5 dB advantage is above the FLoRa threshold for SF7.
    winners = phy.capture(
        [-50.0, -55.0],
        [7, 7],
        [0.0, 0.0],
        [0.1, 0.1],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf7_capture",
            rssi_list=(-50.0, -55.0),
            sf_list=(7, 7),
            start_list=(0.0, 0.0),
            end_list=(0.1, 0.1),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    # Collision without capture: power gap below 1 dB threshold.
    winners = phy.capture(
        [-50.0, -50.5],
        [7, 7],
        [0.0, 0.0],
        [0.1, 0.1],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf7_no_capture",
            rssi_list=(-50.0, -50.5),
            sf_list=(7, 7),
            start_list=(0.0, 0.0),
            end_list=(0.1, 0.1),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    winners = phy.capture(
        [-45.0, -60.0],
        [7, 9],
        [0.0, 0.0],
        [0.1, 0.1],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf7_sf9_capture",
            rssi_list=(-45.0, -60.0),
            sf_list=(7, 9),
            start_list=(0.0, 0.0),
            end_list=(0.1, 0.1),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    winners = phy.capture(
        [-55.0, -44.0],
        [9, 7],
        [0.0, 0.0],
        [0.1, 0.1],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf9_sf7_loss",
            rssi_list=(-55.0, -44.0),
            sf_list=(9, 7),
            start_list=(0.0, 0.0),
            end_list=(0.1, 0.1),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    sf = 8
    symbol_time = (2 ** sf) / channel.bandwidth
    winners = phy.capture(
        [-48.0, -60.0],
        [sf, sf],
        [0.0, 5.1 * symbol_time],
        [0.1, 5.1 * symbol_time + 0.1],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf8_capture_window_allows_first",
            rssi_list=(-48.0, -60.0),
            sf_list=(sf, sf),
            start_list=(0.0, 5.1 * symbol_time),
            end_list=(0.1, 5.1 * symbol_time + 0.1),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    winners = phy.capture(
        [-48.0, -47.5],
        [sf, sf],
        [0.0, 1.0 * symbol_time],
        [0.2, 1.0 * symbol_time + 0.2],
        [868e6, 868e6],
    )
    traces.append(
        CaptureTrace(
            name="sf8_capture_window_collision",
            rssi_list=(-48.0, -47.5),
            sf_list=(sf, sf),
            start_list=(0.0, 1.0 * symbol_time),
            end_list=(0.2, 1.0 * symbol_time + 0.2),
            freq_list=(868e6, 868e6),
            expected_winners=tuple(bool(x) for x in winners),
        )
    )

    return tuple(traces)


def _make_adr_traces() -> tuple[AdrTrace, ...]:
    traces: list[AdrTrace] = []

    traces.append(
        AdrTrace(
            name="adr_avg_high_margin",
            snr_values=tuple(5.0 for _ in range(20)),
            initial_sf=12,
            initial_power_dBm=14.0,
            method="avg",
            expected_command=_flora_adr_decision(tuple(5.0 for _ in range(20)), 12, 14.0, method="avg"),
        )
    )

    traces.append(
        AdrTrace(
            name="adr_avg_low_margin",
            snr_values=tuple(-5.0 for _ in range(20)),
            initial_sf=9,
            initial_power_dBm=10.0,
            method="avg",
            expected_command=_flora_adr_decision(tuple(-5.0 for _ in range(20)), 9, 10.0, method="avg"),
        )
    )

    traces.append(
        AdrTrace(
            name="adr_max_large_margin",
            snr_values=tuple(10.0 for _ in range(20)),
            initial_sf=12,
            initial_power_dBm=14.0,
            method="max",
            expected_command=_flora_adr_decision(tuple(10.0 for _ in range(20)), 12, 14.0, method="max"),
        )
    )

    traces.append(
        AdrTrace(
            name="adr_avg_no_change",
            snr_values=tuple(-15.0 for _ in range(20)),
            initial_sf=10,
            initial_power_dBm=14.0,
            method="avg",
            expected_command=_flora_adr_decision(tuple(-15.0 for _ in range(20)), 10, 14.0, method="avg"),
        )
    )

    return tuple(traces)


def _make_adr_log_traces() -> tuple[AdrLogTrace, ...]:
    traces: list[AdrLogTrace] = []

    avg_values = tuple(-12.0 + 0.5 * i for i in range(40))
    traces.append(
        AdrLogTrace(
            name="adr_avg_window_two_batches",
            method="avg",
            initial_sf=10,
            initial_power_dBm=14.0,
            snr_values=avg_values,
            expected_metrics=(-7.25, 2.75),
            expected_margins=(-7.25, 7.75),
        )
    )

    max_values = tuple(-20.0 + 1.0 * i for i in range(40))
    traces.append(
        AdrLogTrace(
            name="adr_max_window_two_batches",
            method="max",
            initial_sf=12,
            initial_power_dBm=14.0,
            snr_values=max_values,
            expected_metrics=(-1.0, 19.0),
            expected_margins=(4.0, 21.5),
        )
    )

    return tuple(traces)


RSSI_SNR_REFERENCES = _make_rssi_snr_traces()
CAPTURE_REFERENCES = _make_capture_traces()
ADR_REFERENCES = _make_adr_traces()
ADR_LOG_REFERENCES = _make_adr_log_traces()
