"""Helper utilities to build deterministic long range scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

from ..launcher import Channel, Simulator

# Target a 24 km x 24 km deployment area to accommodate 12 km links.
LONG_RANGE_AREA_SIZE: float = 24_000.0
LONG_RANGE_DISTANCES: List[float] = [
    11_000.0,
    10_800.0,
    10_000.0,
    9_000.0,
    8_000.0,
    7_000.0,
    6_000.0,
    5_000.0,
    4_000.0,
]
LONG_RANGE_SPREADING_FACTORS: List[int] = [12, 12, 12, 11, 11, 10, 10, 9, 9]
LONG_RANGE_BANDWIDTHS: tuple[int, int, int] = (125_000, 250_000, 500_000)

# Distances (km) used as reference to interpolate new link budgets. The values
# align with the documentation summary produced from empirical presets.
_REFERENCE_DISTANCE_KM: Dict[str, float] = {
    "rural_long_range": 10.0,
    "flora": 12.0,
    "flora_hata": 12.0,
    "very_long_range": 15.0,
}


@dataclass(frozen=True)
class LongRangeParameters:
    """Hardware assumptions used to stabilise long range simulations."""

    tx_power_dBm: float
    tx_antenna_gain_dB: float
    rx_antenna_gain_dB: float
    cable_loss_dB: float
    packet_interval_s: float = 1200.0
    packets_per_node: int = 8
    shadowing_std_dB: float = 0.0
    distances: tuple[float, ...] | None = None
    spreading_factors: tuple[int, ...] | None = None
    area_size_m: float | None = None


@dataclass(frozen=True)
class SuggestedLongRange:
    """Result container returned by :func:`suggest_parameters`."""

    parameters: LongRangeParameters
    environment: str
    reference_presets: Tuple[str, str]
    interpolation_factor: float
    max_distance_km: float
    area_km2: float


LONG_RANGE_RECOMMENDATIONS: Dict[str, LongRangeParameters] = {
    "flora": LongRangeParameters(
        tx_power_dBm=23.0,
        tx_antenna_gain_dB=16.0,
        rx_antenna_gain_dB=16.0,
        cable_loss_dB=0.5,
    ),
    "flora_hata": LongRangeParameters(
        tx_power_dBm=23.0,
        tx_antenna_gain_dB=16.0,
        rx_antenna_gain_dB=16.0,
        cable_loss_dB=0.5,
    ),
    "rural_long_range": LongRangeParameters(
        tx_power_dBm=16.0,
        tx_antenna_gain_dB=6.0,
        rx_antenna_gain_dB=6.0,
        cable_loss_dB=0.5,
    ),
    "very_long_range": LongRangeParameters(
        tx_power_dBm=27.0,
        tx_antenna_gain_dB=19.0,
        rx_antenna_gain_dB=19.0,
        cable_loss_dB=0.5,
        distances=(
            15_000.0,
            13_500.0,
            12_000.0,
            10_800.0,
            10_000.0,
            9_000.0,
            8_000.0,
            7_000.0,
            6_000.0,
            5_000.0,
            4_000.0,
        ),
        spreading_factors=(12, 12, 12, 12, 11, 11, 10, 10, 9, 9, 9),
        area_size_m=32_000.0,
    ),
}


def _loss_model(preset: str) -> str:
    return "hata" if preset == "flora_hata" else "lognorm"


def _preset_distances(params: LongRangeParameters) -> tuple[float, ...]:
    return params.distances or tuple(LONG_RANGE_DISTANCES)


def _preset_spreading_factors(params: LongRangeParameters) -> tuple[int, ...]:
    return params.spreading_factors or tuple(LONG_RANGE_SPREADING_FACTORS)


def _preset_area_size(params: LongRangeParameters) -> float:
    return params.area_size_m or LONG_RANGE_AREA_SIZE


def _reference_points() -> List[tuple[float, str, LongRangeParameters]]:
    """Return unique interpolation anchors sorted by reference distance."""

    points: Dict[float, tuple[str, LongRangeParameters]] = {}
    for name, params in LONG_RANGE_RECOMMENDATIONS.items():
        distance = _REFERENCE_DISTANCE_KM.get(name)
        if distance is None:
            continue
        # Preserve the first preset registered for a given distance (flora_hata
        # for 12 km, rural_long_range for 10 km, very_long_range for 15 km).
        points.setdefault(distance, (name, params))
    return sorted(((dist, preset, params) for dist, (preset, params) in points.items()), key=lambda item: item[0])


def _interpolate(value: float, lo: float, hi: float, lo_val: float, hi_val: float) -> float:
    if math.isclose(hi, lo):
        return lo_val
    ratio = (value - lo) / (hi - lo)
    return lo_val + ratio * (hi_val - lo_val)


def suggest_parameters(area_km2: float, max_distance_km: float | None = None) -> SuggestedLongRange:
    """Suggest :class:`LongRangeParameters` for the requested coverage.

    The recommendation linearly interpolates the presets declared in
    :data:`LONG_RANGE_RECOMMENDATIONS` using ``_REFERENCE_DISTANCE_KM`` as
    anchors. When the target distance sits outside of the known bounds, the
    closest preset is returned. Distances are scaled from
    :data:`LONG_RANGE_DISTANCES` to maintain the original node layout.
    """

    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive")
    if max_distance_km is not None and max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive when provided")

    side_km = math.sqrt(area_km2)
    if max_distance_km is None:
        # Use half the side length so that the farthest node remains within the
        # target square area.
        max_distance_km = side_km / 2.0

    anchors = _reference_points()
    if not anchors:
        raise RuntimeError("No reference presets defined for long range suggestions")

    lo_dist, lo_name, lo_params = anchors[0]
    hi_dist, hi_name, hi_params = anchors[-1]

    for dist, name, params in anchors:
        if max_distance_km <= dist:
            hi_dist, hi_name, hi_params = dist, name, params
            break
        lo_dist, lo_name, lo_params = dist, name, params

    if max_distance_km <= anchors[0][0]:
        factor = 0.0
        ref_pair = (lo_name, lo_name)
    elif max_distance_km >= anchors[-1][0]:
        factor = 1.0
        ref_pair = (hi_name, hi_name)
    else:
        factor = (max_distance_km - lo_dist) / (hi_dist - lo_dist)
        ref_pair = (lo_name, hi_name)

    tx_power = _interpolate(max_distance_km, lo_dist, hi_dist, lo_params.tx_power_dBm, hi_params.tx_power_dBm)
    tx_gain = _interpolate(max_distance_km, lo_dist, hi_dist, lo_params.tx_antenna_gain_dB, hi_params.tx_antenna_gain_dB)
    rx_gain = _interpolate(max_distance_km, lo_dist, hi_dist, lo_params.rx_antenna_gain_dB, hi_params.rx_antenna_gain_dB)
    cable_loss = _interpolate(max_distance_km, lo_dist, hi_dist, lo_params.cable_loss_dB, hi_params.cable_loss_dB)

    base_distances = tuple(LONG_RANGE_DISTANCES)
    max_base_distance = max(base_distances)
    if max_base_distance <= 0:
        raise RuntimeError("Invalid long range distance reference")
    scale = (max_distance_km * 1_000.0) / max_base_distance
    scaled_distances = tuple(distance * scale for distance in base_distances)

    side_km = max(side_km, max_distance_km * 2.0)
    area_size_m = side_km * 1_000.0

    params = LongRangeParameters(
        tx_power_dBm=tx_power,
        tx_antenna_gain_dB=tx_gain,
        rx_antenna_gain_dB=rx_gain,
        cable_loss_dB=cable_loss,
        distances=scaled_distances,
        spreading_factors=tuple(LONG_RANGE_SPREADING_FACTORS),
        area_size_m=area_size_m,
    )

    # Choose the environment closest to the target distance.
    if factor <= 0.5:
        environment = lo_name
    else:
        environment = hi_name

    return SuggestedLongRange(
        parameters=params,
        environment=environment,
        reference_presets=ref_pair,
        interpolation_factor=factor,
        max_distance_km=max_distance_km,
        area_km2=side_km**2,
    )


def create_long_range_channels(preset: str) -> List[Channel]:
    """Return channels tuned for large area validation."""

    if preset not in LONG_RANGE_RECOMMENDATIONS:
        raise ValueError(f"Unknown long range preset: {preset}")
    params = LONG_RANGE_RECOMMENDATIONS[preset]
    channels: List[Channel] = []
    for bandwidth in LONG_RANGE_BANDWIDTHS:
        channel = Channel(environment=preset, flora_loss_model=_loss_model(preset))
        channel.shadowing_std = params.shadowing_std_dB
        channel.bandwidth = float(bandwidth)
        channel.tx_antenna_gain_dB = params.tx_antenna_gain_dB
        channel.rx_antenna_gain_dB = params.rx_antenna_gain_dB
        channel.cable_loss_dB = params.cable_loss_dB

        # LoRaWAN coverage at 10–15 km hinges on the simulator being able to
        # detect signals close to the theoretical sensitivity limits.  FLoRa
        # exposes these limits via ``Channel.FLORA_SENSITIVITY`` but the
        # default channel constructor keeps a conservative −90 dBm energy
        # threshold.  The long range presets therefore ended up discarding
        # every SF12 transmission as "NoCoverage", driving the PDR to zero.
        #
        # Align the detection logic with the underlying FLoRa tables so that
        # the gateway accepts any frame that is within the published
        # sensitivity budget for the configured bandwidth.
        available = [
            Channel.FLORA_SENSITIVITY[sf][int(bandwidth)]
            for sf in Channel.FLORA_SENSITIVITY
            if int(bandwidth) in Channel.FLORA_SENSITIVITY[sf]
        ]
        if available:
            detection_floor = min(available)
            channel.energy_detection_dBm = detection_floor
            channel.detection_threshold_dBm = detection_floor

        channels.append(channel)
    return channels


def _build_simulator_from_params(
    params: LongRangeParameters,
    preset: str,
    *,
    seed: int,
    packets_per_node: int | None = None,
) -> Simulator:
    distances = _preset_distances(params)
    spreading_factors = _preset_spreading_factors(params)
    if len(distances) != len(spreading_factors):
        raise ValueError("Distances and spreading factors must have matching lengths")

    channels = create_long_range_channels(preset)
    for channel in channels:
        channel.shadowing_std = params.shadowing_std_dB
        channel.tx_antenna_gain_dB = params.tx_antenna_gain_dB
        channel.rx_antenna_gain_dB = params.rx_antenna_gain_dB
        channel.cable_loss_dB = params.cable_loss_dB

    # ``packets_per_node`` explicitly allows ``0`` to request an unlimited run,
    # mirroring :class:`Simulator`'s ``packets_to_send`` parameter.  The previous
    # implementation used ``packets_per_node or ...`` which accidentally
    # replaced ``0`` with the default preset value, making it impossible to
    # disable the packet cap from the helper API.
    packet_budget = (
        packets_per_node if packets_per_node is not None else params.packets_per_node
    )

    detection_floor = min(ch.energy_detection_dBm for ch in channels)

    simulator = Simulator(
        num_nodes=len(distances),
        num_gateways=1,
        area_size=_preset_area_size(params),
        transmission_mode="Periodic",
        packet_interval=params.packet_interval_s,
        packets_to_send=packet_budget,
        mobility=False,
        seed=seed,
        flora_mode=True,
        channels=channels,
        energy_detection_dBm=detection_floor,
        detection_threshold_dBm=detection_floor,
    )
    configure_long_range_nodes(simulator, params)
    return simulator


def configure_long_range_nodes(sim: Simulator, params: LongRangeParameters) -> None:
    """Deterministically place nodes on the x axis and assign SF/BW pairs."""

    distances = _preset_distances(params)
    spreading_factors = _preset_spreading_factors(params)
    if len(distances) != len(spreading_factors):
        raise ValueError("Distances and spreading factors must have matching lengths")
    if len(sim.nodes) != len(distances):
        raise ValueError(
            "Long range scenario expects exactly"
            f" {len(distances)} nodes"
        )
    gateway = sim.gateways[0]
    center_x = gateway.x
    center_y = gateway.y
    channels = sim.multichannel.channels
    for idx, node in enumerate(sim.nodes):
        node.x = center_x + distances[idx]
        node.y = center_y
        node.sf = spreading_factors[idx]
        node.tx_power = params.tx_power_dBm
        node.channel = channels[idx % len(channels)]
        node.chmask = 1 << (idx % len(channels))


def build_long_range_simulator(
    preset: str,
    *,
    seed: int = 2,
    packets_per_node: int | None = None,
) -> Simulator:
    """Create a :class:`Simulator` configured for the large area scenario."""

    if preset not in LONG_RANGE_RECOMMENDATIONS:
        raise ValueError(f"Unknown long range preset: {preset}")
    params = LONG_RANGE_RECOMMENDATIONS[preset]
    return _build_simulator_from_params(
        params,
        preset,
        seed=seed,
        packets_per_node=packets_per_node,
    )


def build_simulator_from_suggestion(
    suggestion: SuggestedLongRange,
    *,
    seed: int = 2,
) -> Simulator:
    """Instantiate a simulator using :func:`suggest_parameters` output."""

    return _build_simulator_from_params(
        suggestion.parameters,
        suggestion.environment,
        seed=seed,
        packets_per_node=suggestion.parameters.packets_per_node,
    )
