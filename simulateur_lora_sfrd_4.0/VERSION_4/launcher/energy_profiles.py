from dataclasses import dataclass

DEFAULT_TX_CURRENT_MAP_A: dict[float, float] = {
    2.0: 0.02,  # ~20 mA
    5.0: 0.027,  # ~27 mA
    8.0: 0.035,  # ~35 mA
    11.0: 0.045,  # ~45 mA
    14.0: 0.060,  # ~60 mA
    17.0: 0.10,  # ~100 mA
    20.0: 0.12,  # ~120 mA
}

@dataclass(frozen=True)
class EnergyProfile:
    """Energy consumption parameters for a LoRa node."""
    voltage_v: float = 3.3
    sleep_current_a: float = 1e-6
    rx_current_a: float = 11e-3
    process_current_a: float = 0.0
    rx_window_duration: float = 0.0
    tx_current_map_a: dict[float, float] | None = None

    def get_tx_current(self, power_dBm: float) -> float:
        """Return TX current for the closest power value in the mapping."""
        if not self.tx_current_map_a:
            return 0.0
        key = min(self.tx_current_map_a.keys(), key=lambda k: abs(k - power_dBm))
        return self.tx_current_map_a[key]


# Default profile based on the FLoRa model (OMNeT++)
FLORA_PROFILE = EnergyProfile(tx_current_map_a=DEFAULT_TX_CURRENT_MAP_A)

# Example of a lower power transceiver profile
LOW_POWER_TX_MAP_A: dict[float, float] = {
    2.0: 0.015,
    5.0: 0.022,
    8.0: 0.029,
    11.0: 0.040,
    14.0: 0.055,
}

LOW_POWER_PROFILE = EnergyProfile(rx_current_a=7e-3, tx_current_map_a=LOW_POWER_TX_MAP_A)

# ------------------------------------------------------------------
# Profile registry helpers
# ------------------------------------------------------------------

PROFILES: dict[str, EnergyProfile] = {
    "flora": FLORA_PROFILE,
    "low_power": LOW_POWER_PROFILE,
}


def register_profile(name: str, profile: EnergyProfile) -> None:
    """Register a named energy profile."""
    PROFILES[name.lower()] = profile


def get_profile(name: str) -> EnergyProfile:
    """Retrieve a named energy profile."""
    key = name.lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown energy profile: {name}")
    return PROFILES[key]
