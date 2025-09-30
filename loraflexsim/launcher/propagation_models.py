import math
import random

class LogDistanceShadowing:
    """Simple log-distance model with log-normal shadowing."""

    ENV_PARAMS = {
        "urban": (2.08, 3.57, 127.41, 40.0),
        "suburban": (2.32, 7.08, 128.95, 1000.0),
        "rural": (2.0, 2.0, 113.0, 1.0),
        "flora": (2.7, 3.57, 127.41, 40.0),
    }

    def __init__(
        self,
        frequency_hz: float = 868e6,
        *,
        path_loss_exp: float = 2.7,
        shadowing_std: float = 6.0,
        path_loss_d0: float | None = None,
        reference_distance: float = 1.0,
        environment: str | None = None,
    ) -> None:
        if environment is not None:
            env = environment.lower()
            if env not in self.ENV_PARAMS:
                raise ValueError(f"Unknown environment: {environment}")
            path_loss_exp, shadowing_std, path_loss_d0, reference_distance = self.ENV_PARAMS[env]
        if path_loss_d0 is None:
            freq_mhz = frequency_hz / 1e6
            path_loss_d0 = 32.45 + 20 * math.log10(freq_mhz) - 60.0
        self.frequency_hz = frequency_hz
        self.path_loss_exp = path_loss_exp
        self.shadowing_std = shadowing_std
        self.path_loss_d0 = path_loss_d0
        self.reference_distance = reference_distance

    def path_loss(self, distance: float) -> float:
        if distance <= 0:
            return 0.0
        d = max(distance, 1.0)
        loss = self.path_loss_d0 + 10 * self.path_loss_exp * math.log10(
            d / self.reference_distance
        )
        if self.shadowing_std > 0:
            loss += random.gauss(0.0, self.shadowing_std)
        return loss


def multipath_fading_db(taps: int = 1) -> float:
    """Return a multipath fading value in dB based on Rayleigh paths."""
    taps = int(taps)
    if taps <= 1:
        return 0.0
    i = sum(random.gauss(0.0, 1.0) for _ in range(taps))
    q = sum(random.gauss(0.0, 1.0) for _ in range(taps))
    amp = math.sqrt(i * i + q * q) / math.sqrt(taps)
    return 20 * math.log10(max(amp, 1e-12))


class CompletePropagation(LogDistanceShadowing):
    """More detailed propagation model including fast fading and calibrated noise."""

    SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
    # Sensitivity values from the Semtech SX1272/73 datasheet (also used by FLoRa)
    SEMTECH_SENSITIVITY = {
        6: {125000: -121, 250000: -118, 500000: -111},
        7: {125000: -124, 250000: -122, 500000: -116},
        8: {125000: -127, 250000: -125, 500000: -119},
        9: {125000: -130, 250000: -128, 500000: -122},
        10: {125000: -133, 250000: -130, 500000: -125},
        11: {125000: -135, 250000: -132, 500000: -128},
        12: {125000: -137, 250000: -135, 500000: -129},
    }

    def __init__(
        self,
        frequency_hz: float = 868e6,
        *,
        path_loss_exp: float = 2.7,
        shadowing_std: float = 6.0,
        path_loss_d0: float | None = None,
        reference_distance: float = 1.0,
        fast_fading_std: float = 0.0,
        multipath_taps: int = 1,
        noise_figure_dB: float = 6.0,
        temperature_K: float = 290.0,
        environment: str | None = None,
    ) -> None:
        super().__init__(
            frequency_hz,
            path_loss_exp=path_loss_exp,
            shadowing_std=shadowing_std,
            path_loss_d0=path_loss_d0,
            reference_distance=reference_distance,
            environment=environment,
        )
        self.fast_fading_std = fast_fading_std
        self.multipath_taps = int(multipath_taps)
        self.noise_figure_dB = noise_figure_dB
        self.temperature_K = temperature_K

    @staticmethod
    def thermal_noise_dBm(bandwidth: float, temperature_K: float = 290.0) -> float:
        k = 1.38064852e-23
        return 10 * math.log10(k * temperature_K * bandwidth) + 30

    def noise_floor_dBm(self, bandwidth: float) -> float:
        base = self.thermal_noise_dBm(bandwidth, self.temperature_K)
        return base + self.noise_figure_dB

    def rssi(self, tx_power_dBm: float, distance: float) -> float:
        rssi = tx_power_dBm - self.path_loss(distance)
        if self.fast_fading_std > 0:
            rssi += random.gauss(0.0, self.fast_fading_std)
        if self.multipath_taps > 1:
            rssi += multipath_fading_db(self.multipath_taps)
        return rssi

    def sensitivity_table(self, bandwidth: float) -> dict[int, float]:
        bw = int(bandwidth)
        if bw in (125000, 250000, 500000):
            return {sf: vals[bw] for sf, vals in self.SEMTECH_SENSITIVITY.items()}
        floor = self.noise_floor_dBm(bandwidth)
        return {sf: floor + snr for sf, snr in self.SNR_THRESHOLDS.items()}
