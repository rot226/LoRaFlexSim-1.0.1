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

    densities_per_km2: Sequence[float] = (0.1, 0.5, 1.0)
    radius_m: int = 1000
    duration_s: int = 3600
    payload_bytes: int = 20
    traffic_model: str = "poisson"


@dataclass(frozen=True)
class SNIRConfig:
    """Paramètres SNIR."""

    enabled: bool = True
    capture_threshold_db: float = 6.0
    thermal_noise_dbm: float = -174.0


@dataclass(frozen=True)
class QoSConfig:
    """Paramètres QoS."""

    clusters: Sequence[str] = ("gold", "silver", "bronze")
    proportions: Sequence[float] = (0.2, 0.3, 0.5)


@dataclass(frozen=True)
class RLConfig:
    """Paramètres RL."""

    window_w: int = 20
    warmup: int = 5
    lambda_energy: float = 0.1


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


DEFAULT_CONFIG = AppConfig()
