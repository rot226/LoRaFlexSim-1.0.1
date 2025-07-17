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
    process_current_a: float = 5e-3
    rx_window_duration: float = 0.1
    tx_current_map_a: dict[float, float] | None = None

    def get_tx_current(self, power_dBm: float) -> float:
        """Return TX current for the closest power value in the mapping."""
        if not self.tx_current_map_a:
            return 0.0
        key = min(self.tx_current_map_a.keys(), key=lambda k: abs(k - power_dBm))
        return self.tx_current_map_a[key]


# Default profile based on the FLoRa model (OMNeT++)
FLORA_PROFILE = EnergyProfile(tx_current_map_a=DEFAULT_TX_CURRENT_MAP_A)
