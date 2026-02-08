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
    snir_threshold_min_db: float = 3.0
    snir_threshold_max_db: float = 6.0
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
    traffic_coeff_scale: float = 0.75
    traffic_coeff_clamp_min: float = 0.4
    traffic_coeff_clamp_max: float = 2.5
    traffic_coeff_clamp_enabled: bool = True
    window_delay_enabled: bool = True
    window_delay_range_s: float = 5.0
    capture_probability: float = 0.2
    congestion_coeff: float = 1.0
    congestion_coeff_base: float = 0.28
    congestion_coeff_growth: float = 0.3
    congestion_coeff_max: float = 0.3
    link_success_min_ratio: float = 0.02
    network_load_min: float = 0.75
    network_load_max: float = 1.5
    collision_size_min: float = 0.75
    collision_size_under_max: float = 1.25
    collision_size_over_max: float = 1.5
    collision_size_factor: float | None = None
    lambda_collision_base: float = 0.12
    lambda_collision_min: float = 0.06
    lambda_collision_max: float = 0.6
    lambda_collision_overload_scale: float = 0.35
    reward_floor: float | None = None
    floor_on_zero_success: bool = False
    max_penalty_ratio: float = 0.8
    shadowing_sigma_db: float | None = None


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

STEP2_SAFE_CONFIG = Step2Config(
    capture_probability=0.25,
    traffic_coeff_clamp_min=0.6,
    traffic_coeff_clamp_max=2.2,
    traffic_coeff_clamp_enabled=True,
    network_load_min=0.8,
    network_load_max=1.4,
    collision_size_min=0.85,
    collision_size_under_max=1.1,
    collision_size_over_max=1.4,
    reward_floor=0.05,
    floor_on_zero_success=True,
    max_penalty_ratio=0.5,
    shadowing_sigma_db=8.0,
)

STEP2_SUPER_SAFE_CONFIG = Step2Config(
    capture_probability=0.3,
    traffic_coeff_clamp_min=0.7,
    traffic_coeff_clamp_max=2.0,
    traffic_coeff_clamp_enabled=True,
    network_load_min=0.9,
    network_load_max=1.2,
    collision_size_min=0.9,
    collision_size_under_max=1.0,
    collision_size_over_max=1.2,
    reward_floor=0.06,
    floor_on_zero_success=True,
    max_penalty_ratio=0.4,
    shadowing_sigma_db=9.0,
)
