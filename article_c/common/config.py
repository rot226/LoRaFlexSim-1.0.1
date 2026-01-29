"""Configuration globale pour l'article C."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


@dataclass(frozen=True)
class RadioConfig:
    """Paramètres radio."""

    bandwidth_khz: int = 125
    coding_rate: str = "4/5"
    tx_power_dbm: int = 14
    spreading_factors: Sequence[int] = (7, 8, 9, 10, 11, 12)
    channels_hz: Sequence[int] = (868100000, 868300000, 868500000)


@dataclass(frozen=True)
class ScenarioConfig:
    """Paramètres du scénario."""

    network_sizes: Sequence[int] = (50, 100, 150)
    radius_m: int = 1000
    duration_s: int = 3600
    payload_bytes: int = 20
    shadowing_sigma_db: float = 7.0
    shadowing_mean_db: float = 0.0
    traffic_mode: str = "periodic"
    jitter_range: float | None = None


@dataclass(frozen=True)
class SNIRConfig:
    """Paramètres SNIR (bruit en dBm/Hz)."""

    enabled: bool = True
    snir_threshold_db: float = 5.0
    noise_floor_dbm: float = -174.0


@dataclass(frozen=True)
class QoSConfig:
    """Paramètres QoS."""

    clusters: Sequence[str] = ("gold", "silver", "bronze")
    proportions: Sequence[float] = (0.2, 0.3, 0.5)


@dataclass(frozen=True)
class RLConfig:
    """Paramètres RL."""

    window_w: int = 12
    warmup: int = 5
    lambda_energy: float = 0.2
    lambda_collision: float | None = None


@dataclass(frozen=True)
class Step2Config:
    """Paramètres spécifiques à l'étape 2."""

    traffic_mode: str = "poisson"
    jitter_range_s: float = 30.0
    window_duration_s: float = 60.0
    traffic_coeff_min: float = 0.7
    traffic_coeff_max: float = 1.3
    traffic_coeff_enabled: bool = True
    window_delay_enabled: bool = True
    window_delay_range_s: float = 5.0


@dataclass(frozen=True)
class AppConfig:
    """Configuration agrégée."""

    base_dir: Path = BASE_DIR
    results_dir: Path = RESULTS_DIR
    plots_dir: Path = PLOTS_DIR
    radio: RadioConfig = field(default_factory=RadioConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    snir: SNIRConfig = field(default_factory=SNIRConfig)
    qos: QoSConfig = field(default_factory=QoSConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    step2: Step2Config = field(default_factory=Step2Config)


DEFAULT_CONFIG = AppConfig()
