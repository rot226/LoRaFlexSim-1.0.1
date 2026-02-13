"""Configuration globale pour l'article C."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


CLUSTER_CANONICAL_TABLE: dict[str, dict[str, tuple[str, ...] | str]] = {
    "cluster_1": {
        "base_label": "Cluster 1",
        "aliases": ("cluster_1", "gold"),
    },
    "cluster_2": {
        "base_label": "Cluster 2",
        "aliases": ("cluster_2", "silver"),
    },
    "cluster_3": {
        "base_label": "Cluster 3",
        "aliases": ("cluster_3", "bronze"),
    },
    "all": {
        "base_label": "All clusters",
        "aliases": ("all",),
    },
}

ALGORITHM_CANONICAL_TABLE: dict[str, dict[str, tuple[str, ...] | str]] = {
    "adr": {
        "label": "ADR",
        "aliases": ("adr", "adr_pure", "adr-pure", "adr pur"),
    },
    "apra": {
        "label": "APRA",
        "aliases": ("apra", "apra_like", "apra-like"),
    },
    "aimi": {
        "label": "Aimi",
        "aliases": ("aimi", "aimi_like", "aimi-like"),
    },
    "loba": {
        "label": "LoBa",
        "aliases": ("loba", "lo_ba", "lora_baseline", "lorawan_baseline"),
    },
    "mixra_h": {
        "label": "MixRA-H",
        "aliases": ("mixra_h", "mixra_hybrid", "mixra-h", "mixra h"),
    },
    "mixra_opt": {
        "label": "MixRA-Opt",
        "aliases": ("mixra_opt", "mixra_optimal", "mixraopt", "mixra-opt"),
    },
    "ucb1_sf": {
        "label": "UCB1-SF",
        "aliases": ("ucb1_sf", "ucb1", "ucb1-sf", "ucb1 sf"),
    },
}

SNIR_CANONICAL_TABLE: dict[str, dict[str, tuple[str, ...] | str]] = {
    "snir_on": {
        "label": "SNIR on",
        "aliases": ("snir_on", "on", "true", "1", "yes"),
    },
    "snir_off": {
        "label": "SNIR off",
        "aliases": ("snir_off", "off", "false", "0", "no"),
    },
    "snir_unknown": {
        "label": "SNIR unknown",
        "aliases": ("snir_unknown", "unknown", "n/a", "na"),
    },
}


def _build_alias_lookup(
    canonical_table: dict[str, dict[str, tuple[str, ...] | str]],
) -> dict[str, str]:
    alias_lookup: dict[str, str] = {}
    for canonical_id, config in canonical_table.items():
        alias_lookup[canonical_id.strip().lower()] = canonical_id
        for alias in config.get("aliases", ()):  # type: ignore[arg-type]
            alias_lookup[str(alias).strip().lower()] = canonical_id
    return alias_lookup


ALGORITHM_ALIAS_TO_CANONICAL = _build_alias_lookup(ALGORITHM_CANONICAL_TABLE)
SNIR_ALIAS_TO_CANONICAL = _build_alias_lookup(SNIR_CANONICAL_TABLE)
CLUSTER_ALIAS_TO_CANONICAL = _build_alias_lookup(CLUSTER_CANONICAL_TABLE)


def normalize_algorithm(value: object, default: str | None = None) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return default
    return ALGORITHM_ALIAS_TO_CANONICAL.get(text, default)


def normalize_snir_mode(value: object, default: str | None = None) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return default
    return SNIR_ALIAS_TO_CANONICAL.get(text, default)


def normalize_cluster(value: object, default: str = "all") -> str:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return default
    return CLUSTER_ALIAS_TO_CANONICAL.get(text, str(value).strip())


def algorithm_label(value: object, default: str = "unknown") -> str:
    canonical = normalize_algorithm(value)
    if canonical is None:
        text = str(value).strip()
        return text if text else default
    return str(ALGORITHM_CANONICAL_TABLE[canonical]["label"])


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

    clusters: Sequence[str] = ("cluster_1", "cluster_2", "cluster_3")
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
    traffic_coeff_clamp_min: float = 0.55
    traffic_coeff_clamp_max: float = 1.9
    traffic_coeff_clamp_enabled: bool = True
    clamped_nodes_ratio_threshold: float = 0.7
    clamped_load_adjust_min_scale: float = 0.55
    window_delay_enabled: bool = True
    window_delay_range_s: float = 5.0
    # Facteur de sécurité de capacité Tx par fenêtre.
    # Plus élevé => moins de paquets admissibles, collisions réduites.
    tx_window_safety_factor: float = 4.0
    capture_probability: float = 0.28
    congestion_coeff: float = 1.0
    congestion_coeff_base: float = 0.28
    congestion_coeff_growth: float = 0.3
    congestion_coeff_max: float = 0.3
    # Garde-fou post-congestion/lien: part minimale de succès conservée.
    # Calibré pour conserver un success_rate non-trivial à N=80.
    link_success_min_ratio: float = 0.65
    network_load_min: float = 0.6
    network_load_max: float = 1.65
    # Calibrage facteur de taille collision (ratio N / N_ref).
    # - collision_size_min: borne basse quand N <= N_ref.
    # - collision_size_under_max: borne haute en sous-charge.
    # - collision_size_over_max: borne haute en surcharge.
    collision_size_min: float = 0.72
    collision_size_under_max: float = 1.02
    collision_size_over_max: float = 1.45
    collision_size_factor: float | None = None
    # Coefficients explicites du calcul collision_norm pour tuning.
    collision_norm_airtime_exp: float = 1.1
    collision_norm_congestion_gain: float = 0.45
    collision_norm_size_exp: float = 0.7
    collision_norm_failure_exp: float = 0.55
    collision_norm_offset: float = 0.08
    lambda_collision_base: float = 0.1
    lambda_collision_min: float = 0.06
    lambda_collision_max: float = 0.6
    lambda_collision_overload_scale: float = 0.35
    reward_floor: float | None = None
    floor_on_zero_success: bool = False
    zero_success_quality_bonus_factor: float = 0.0
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
    capture_probability=0.32,
    traffic_coeff_clamp_min=0.65,
    traffic_coeff_clamp_max=1.8,
    traffic_coeff_clamp_enabled=True,
    network_load_min=0.65,
    network_load_max=1.45,
    collision_size_min=0.75,
    collision_size_under_max=1.0,
    collision_size_over_max=1.35,
    reward_floor=0.05,
    floor_on_zero_success=True,
    zero_success_quality_bonus_factor=0.05,
    max_penalty_ratio=0.5,
    shadowing_sigma_db=8.0,
)

STEP2_SUPER_SAFE_CONFIG = Step2Config(
    capture_probability=0.36,
    traffic_coeff_clamp_min=0.75,
    traffic_coeff_clamp_max=1.6,
    traffic_coeff_clamp_enabled=True,
    network_load_min=0.75,
    network_load_max=1.3,
    collision_size_min=0.8,
    collision_size_under_max=1.0,
    collision_size_over_max=1.3,
    reward_floor=0.06,
    floor_on_zero_success=True,
    zero_success_quality_bonus_factor=0.05,
    max_penalty_ratio=0.4,
    shadowing_sigma_db=9.0,
)
