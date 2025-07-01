from dataclasses import dataclass

@dataclass(frozen=True)
class EnergyProfile:
    """Energy consumption parameters for a LoRa node."""
    voltage_v: float = 3.3
    sleep_current_a: float = 1e-6
    rx_current_a: float = 11e-3
    process_current_a: float = 5e-3
    rx_window_duration: float = 0.1


# Default profile based on the FLoRa model (OMNeT++)
FLORA_PROFILE = EnergyProfile()
