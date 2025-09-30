"""Predefined simulation scenarios for LoRaFlexSim."""

from .long_range import (
    LONG_RANGE_AREA_SIZE,
    LONG_RANGE_BANDWIDTHS,
    LONG_RANGE_DISTANCES,
    LONG_RANGE_RECOMMENDATIONS,
    LONG_RANGE_SPREADING_FACTORS,
    LongRangeParameters,
    SuggestedLongRange,
    build_long_range_simulator,
    build_simulator_from_suggestion,
    configure_long_range_nodes,
    create_long_range_channels,
    suggest_parameters,
)

__all__ = [
    "LONG_RANGE_AREA_SIZE",
    "LONG_RANGE_BANDWIDTHS",
    "LONG_RANGE_DISTANCES",
    "LONG_RANGE_RECOMMENDATIONS",
    "LONG_RANGE_SPREADING_FACTORS",
    "LongRangeParameters",
    "SuggestedLongRange",
    "build_long_range_simulator",
    "build_simulator_from_suggestion",
    "configure_long_range_nodes",
    "create_long_range_channels",
    "suggest_parameters",
]
