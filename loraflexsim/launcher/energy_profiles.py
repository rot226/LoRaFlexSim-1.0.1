from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

# Conversion of the FLoRa (OMNeT++) ``energyConsumptionParameters.xml`` TX
# currents from milliamperes to amperes.
DEFAULT_TX_CURRENT_MAP_A: dict[float, float] = {
    2.0: 0.024,
    3.0: 0.024,
    4.0: 0.024,
    5.0: 0.025,
    6.0: 0.025,
    7.0: 0.025,
    8.0: 0.025,
    9.0: 0.026,
    10.0: 0.031,
    11.0: 0.032,
    12.0: 0.034,
    13.0: 0.035,
    14.0: 0.044,
}

@dataclass(frozen=True)
class EnergyProfile:
    """Energy consumption parameters for a LoRa node."""

    voltage_v: float = 3.3
    sleep_current_a: float = 1e-7
    rx_current_a: float = 9.7e-3
    listen_current_a: float = 0.0
    process_current_a: float = 0.0
    # Additional radio state modelling parameters
    startup_current_a: float = 0.0
    startup_time_s: float = 0.0
    preamble_current_a: float = 0.0
    preamble_time_s: float = 0.0
    ramp_up_s: float = 0.0
    ramp_down_s: float = 0.0
    rx_window_duration: float = 0.0
    include_transients: bool = True
    tx_current_map_a: dict[float, float] | None = None

    def get_tx_current(self, power_dBm: float) -> float:
        """Return TX current for the closest power value in the mapping."""
        if not self.tx_current_map_a:
            return 0.0
        key = min(self.tx_current_map_a.keys(), key=lambda k: abs(k - power_dBm))
        return self.tx_current_map_a[key]

    def current_for(
        self, state: str, power_dBm: float | None = None
    ) -> float:
        """Return the current drawn when the radio is in ``state``."""

        key = state.lower()
        if key == "sleep":
            return self.sleep_current_a
        if key == "rx":
            return self.rx_current_a
        if key == "listen":
            return (
                self.listen_current_a
                if self.listen_current_a > 0.0
                else self.rx_current_a
            )
        if key == "processing":
            return self.process_current_a
        if key == "startup":
            if not self.include_transients:
                return 0.0
            return self.startup_current_a
        if key == "preamble":
            if not self.include_transients:
                return 0.0
            return self.preamble_current_a
        if key == "tx":
            if power_dBm is None:
                raise ValueError("power_dBm is required for TX energy computation")
            return self.get_tx_current(power_dBm)
        if key == "ramp":
            # Ramp phases use the TX (or listen) current depending on context.
            if not self.include_transients:
                return 0.0
            if self.listen_current_a > 0.0:
                return self.listen_current_a
            if power_dBm is None:
                return self.rx_current_a
            return self.get_tx_current(power_dBm)
        return 0.0

    def energy_for(
        self, state: str, duration_s: float, power_dBm: float | None = None
    ) -> float:
        """Return the energy (J) spent in ``state`` during ``duration_s`` seconds."""

        if duration_s <= 0:
            return 0.0
        current = self.current_for(state, power_dBm)
        return current * self.voltage_v * duration_s

    def enforce_energy(
        self,
        state: str,
        duration_s: float,
        energy_joules: float,
        power_dBm: float | None = None,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-12,
    ) -> float:
        """
        Ensure that the provided ``energy_joules`` matches ``E = V·I·t``.

        The method returns the corrected energy value.  When ``duration_s`` is
        non-positive the original value is returned unchanged.
        """

        if duration_s <= 0:
            return energy_joules
        expected = self.energy_for(state, duration_s, power_dBm)
        if expected == 0.0:
            return 0.0
        if math.isclose(energy_joules, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return energy_joules
        return expected


class EnergyAccumulator:
    """Simple accumulator tracking energy per state."""

    def __init__(self) -> None:
        self._by_state: defaultdict[str, float] = defaultdict(float)

    def add(self, state: str, energy_joules: float) -> None:
        self._by_state[state] += energy_joules

    def get(self, state: str) -> float:
        return self._by_state.get(state, 0.0)

    def total(self) -> float:
        return sum(self._by_state.values())

    def to_dict(self) -> dict[str, float]:
        return dict(self._by_state)


# Default profile derived from the FLoRa (OMNeT++) energy model parameters
# defined in ``flora-master/simulations/energyConsumptionParameters.xml``.
# Currents are expressed in amperes (A).
FLORA_PROFILE = EnergyProfile(
    tx_current_map_a=DEFAULT_TX_CURRENT_MAP_A,
    startup_current_a=1.6e-3,
    startup_time_s=1e-3,
    preamble_current_a=5e-3,
    preamble_time_s=1e-3,
    ramp_up_s=1e-3,
    ramp_down_s=1e-3,
    rx_window_duration=1.0,
    include_transients=False,
)

# Example of a lower power transceiver profile.  Values keep roughly the same
# relative offsets with respect to the FLoRa profile while remaining strictly
# lower for the supported power levels.
LOW_POWER_TX_MAP_A: dict[float, float] = {
    2.0: 0.018,
    5.0: 0.020375,
    8.0: 0.020725,
    11.0: 0.028416,
    14.0: 0.040348,
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
