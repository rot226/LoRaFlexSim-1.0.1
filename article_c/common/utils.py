"""Utilitaires partagés pour l'article C."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from article_c.common.config import DEFAULT_CONFIG


def ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI."""
    parser = argparse.ArgumentParser(description="Outils CLI pour l'article C.")
    snir_defaults = DEFAULT_CONFIG.snir
    step2_defaults = DEFAULT_CONFIG.step2
    parser.add_argument(
        "--seeds_base",
        type=int,
        default=None,
        help="Seed de base déterministe.",
    )
    parser.add_argument(
        "--seed",
        dest="seeds_base",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias de --seeds_base (déprécié).",
    )
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIG.scenario.network_sizes),
        help="Tailles de réseau (nombre de nœuds entiers, ex: 50 100 150).",
    )
    parser.add_argument(
        "--reference-network-size",
        type=int,
        default=None,
        help=(
            "Taille de référence utilisée pour les facteurs de charge "
            "(par défaut: médiane des tailles demandées)."
        ),
    )
    parser.add_argument(
        "--densities",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Alias de --network-sizes (déprécié).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=10,
        help="Nombre de réplications.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Ajoute un timestamp dans les sorties.",
    )
    parser.add_argument(
        "--safe-profile",
        action="store_true",
        help="Active le profil sécurisé pour l'étape 2.",
    )
    parser.add_argument(
        "--auto-safe-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Active/désactive l'application automatique du profil sécurisé "
            "avant la simulation (activé par défaut)."
        ),
    )
    parser.add_argument(
        "--allow-low-success-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Autorise un success_rate global trop faible en étape 2 "
            "(log un avertissement au lieu d'arrêter)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Active le mode strict pour l'étape 2 (arrêt si success_rate trop faible)."
        ),
    )
    parser.add_argument(
        "--plot-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Génère un plot de synthèse avec barres d'erreur.",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=snir_defaults.snir_threshold_db,
        help="Seuil SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-min-db",
        type=float,
        default=snir_defaults.snir_threshold_min_db,
        help="Borne basse de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-max-db",
        type=float,
        default=snir_defaults.snir_threshold_max_db,
        help="Borne haute de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=snir_defaults.noise_floor_dbm,
        help="Bruit thermique (densité en dBm/Hz).",
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default=step2_defaults.traffic_mode,
        choices=("periodic", "poisson"),
        help="Modèle de trafic pour l'étape 2 (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range-s",
        dest="jitter_range_s",
        type=float,
        default=30.0,
        help="Amplitude du jitter pour l'étape 2 (secondes).",
    )
    parser.add_argument(
        "--jitter-range",
        dest="jitter_range_s",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias de --jitter-range-s (déprécié).",
    )
    parser.add_argument(
        "--window-duration-s",
        type=float,
        default=step2_defaults.window_duration_s,
        help="Durée d'une fenêtre de simulation (secondes).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_CONFIG.rl.window_w,
        help="Taille de la fenêtre (W) pour l'étape 2.",
    )
    parser.add_argument(
        "--lambda-collision",
        type=float,
        default=DEFAULT_CONFIG.rl.lambda_collision,
        help=(
            "Poids de pénalisation des collisions (par défaut: dérivé de lambda_energy)."
        ),
    )
    parser.add_argument(
        "--traffic-coeff-min",
        type=float,
        default=step2_defaults.traffic_coeff_min,
        help="Coefficient de trafic minimal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-max",
        type=float,
        default=step2_defaults.traffic_coeff_max,
        help="Coefficient de trafic maximal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-enabled",
        action=argparse.BooleanOptionalAction,
        default=step2_defaults.traffic_coeff_enabled,
        help="Active/désactive la variabilité de trafic par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-scale",
        type=float,
        default=step2_defaults.traffic_coeff_scale,
        help=(
            "Facteur global appliqué à la charge de trafic (ex: 0.7 pour diminuer)."
        ),
    )
    parser.add_argument(
        "--auto-collision-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Active l'ajustement automatique de traffic_coeff_scale et window_duration_s "
            "en cas de collisions élevées."
        ),
    )
    parser.add_argument(
        "--capture-probability",
        type=float,
        default=step2_defaults.capture_probability,
        help="Probabilité de capture lors d'une collision (0 à 1).",
    )
    parser.add_argument(
        "--congestion-coeff",
        type=float,
        default=step2_defaults.congestion_coeff,
        help=(
            "Coefficient multiplicatif appliqué à la probabilité de congestion "
            "(1.0 pour garder la valeur calculée)."
        ),
    )
    parser.add_argument(
        "--congestion-coeff-base",
        type=float,
        default=step2_defaults.congestion_coeff_base,
        help="Coefficient de base de la probabilité de congestion (0 à 1).",
    )
    parser.add_argument(
        "--congestion-coeff-growth",
        type=float,
        default=step2_defaults.congestion_coeff_growth,
        help="Coefficient de croissance de la probabilité de congestion.",
    )
    parser.add_argument(
        "--congestion-coeff-max",
        type=float,
        default=step2_defaults.congestion_coeff_max,
        help="Plafond de probabilité de congestion (0 à 1).",
    )
    parser.add_argument(
        "--network-load-min",
        type=float,
        default=step2_defaults.network_load_min,
        help="Borne minimale du facteur de charge réseau.",
    )
    parser.add_argument(
        "--network-load-max",
        type=float,
        default=step2_defaults.network_load_max,
        help="Borne maximale du facteur de charge réseau.",
    )
    parser.add_argument(
        "--collision-size-min",
        type=float,
        default=step2_defaults.collision_size_min,
        help="Borne minimale du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-under-max",
        type=float,
        default=step2_defaults.collision_size_under_max,
        help="Borne max (sous-charge) du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-over-max",
        type=float,
        default=step2_defaults.collision_size_over_max,
        help="Borne max (surcharge) du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-factor",
        type=float,
        default=step2_defaults.collision_size_factor,
        help=(
            "Facteur de taille appliqué aux collisions (override du calcul "
            "par taille de réseau si fourni)."
        ),
    )
    parser.add_argument(
        "--traffic-coeff-clamp-min",
        type=float,
        default=step2_defaults.traffic_coeff_clamp_min,
        help="Borne minimale du clamp appliqué aux coefficients de trafic.",
    )
    parser.add_argument(
        "--traffic-coeff-clamp-max",
        type=float,
        default=step2_defaults.traffic_coeff_clamp_max,
        help="Borne maximale du clamp appliqué aux coefficients de trafic.",
    )
    parser.add_argument(
        "--traffic-coeff-clamp-enabled",
        action=argparse.BooleanOptionalAction,
        default=step2_defaults.traffic_coeff_clamp_enabled,
        help="Active/désactive le clamp des coefficients de trafic (diagnostic).",
    )
    parser.add_argument(
        "--window-delay-enabled",
        action=argparse.BooleanOptionalAction,
        default=step2_defaults.window_delay_enabled,
        help="Active/désactive le délai aléatoire entre fenêtres.",
    )
    parser.add_argument(
        "--window-delay-range-s",
        type=float,
        default=step2_defaults.window_delay_range_s,
        help="Amplitude du délai aléatoire entre fenêtres (secondes).",
    )
    parser.add_argument(
        "--reward-floor",
        type=float,
        default=step2_defaults.reward_floor,
        help=(
            "Plancher de récompense appliqué dès que success_rate > 0 "
            "(par défaut: plancher implicite selon l'algorithme)."
        ),
    )
    parser.add_argument(
        "--floor-on-zero-success",
        action=argparse.BooleanOptionalAction,
        default=step2_defaults.floor_on_zero_success,
        help=(
            "Applique un plancher minimal même si success_rate == 0 "
            "(utile pour éviter des rewards uniformes en conditions extrêmes)."
        ),
    )
    parser.add_argument(
        "--zero-success-quality-bonus-factor",
        type=float,
        default=step2_defaults.zero_success_quality_bonus_factor,
        help=(
            "Facteur multiplicatif appliqué à weighted_quality comme bonus minimal "
            "quand success_rate == 0."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Nombre de processus worker pour paralléliser les tailles.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les tailles déjà présentes dans aggregated_results.csv.",
    )
    parser.add_argument(
        "--flat-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Écrit les résultats directement dans le répertoire de sortie "
            "(compatibilité avec l'ancien format)."
        ),
    )
    parser.add_argument(
        "--debug-step2",
        action="store_true",
        help="Active les logs détaillés pour l'étape 2.",
    )
    parser.add_argument(
        "--reward-debug",
        action="store_true",
        help=(
            "Active les logs détaillés des composantes de reward "
            "(weighted_quality, collision_penalty, success_term, reward_floor)."
        ),
    )
    parser.add_argument(
        "--reward-alert-level",
        type=str.upper,
        default="WARNING",
        choices=("INFO", "WARNING"),
        help=(
            "Niveau de log pour l'alerte de rewards uniformes (INFO ou WARNING). "
            "Utiliser INFO pour réduire la verbosité."
        ),
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Formats d'export des figures (ex: png,eps).",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help=(
            "Désactive les clamps (plages réseau/collision, coefficients de trafic "
            "et bornes internes)."
        ),
    )
    parser.add_argument(
        "--no-figure-clamp",
        action="store_true",
        help="Désactive le clamp de taille des figures.",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = build_arg_parser()
    return parser.parse_args(argv)


def set_deterministic_seed(seed: int | None) -> int:
    """Initialise les seeds Python et NumPy de manière déterministe."""
    if seed is None:
        seed = random.randint(1, 10**9)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def parse_network_size_list(value: str | Sequence[int]) -> list[int]:
    """Parse une liste de tailles de réseau (nombre de nœuds)."""
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def replication_ids(count: int) -> list[int]:
    """Retourne les identifiants de réplications."""
    return list(range(1, count + 1))


def timestamp_tag(with_timezone: bool = True) -> str:
    """Retourne un timestamp compatible Windows pour les sorties."""
    if with_timezone:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d_%H-%M-%SZ")
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def flatten(values: Iterable[Sequence[float]]) -> list[float]:
    """Aplatit une liste de séquences numériques."""
    return [item for sequence in values for item in sequence]


def generate_traffic_times(
    sent: int,
    *,
    duration_s: float,
    traffic_mode: str,
    jitter_range_s: float,
    rng: random.Random | None = None,
) -> list[float]:
    """Génère des instants de transmission périodiques ou poisson."""

    if sent <= 0:
        return []
    if duration_s <= 0:
        raise ValueError("duration_s doit être positif")

    generator = rng or random
    base_period = duration_s / sent
    mode = traffic_mode.lower()
    times: list[float] = []
    if mode == "periodic":
        times = [idx * base_period for idx in range(sent)]
    elif mode == "poisson":
        current = 0.0
        while current < duration_s:
            current += generator.expovariate(1.0 / base_period)
            if current < duration_s:
                times.append(current)
    else:
        raise ValueError(f"traffic_mode inconnu: {traffic_mode}")

    if jitter_range_s > 0:
        jittered: list[float] = []
        for t in times:
            jitter = generator.uniform(-jitter_range_s, jitter_range_s)
            candidate = t + jitter
            if 0 <= candidate <= duration_s:
                jittered.append(candidate)
        times = sorted(jittered)

    return times


def assign_clusters(
    count: int,
    *,
    rng: random.Random | None = None,
    clusters: Sequence[str] | None = None,
    proportions: Sequence[float] | None = None,
) -> list[str]:
    """Attribue un cluster à chaque nœud selon des proportions configurables."""

    if count <= 0:
        return []
    generator = rng or random
    if clusters is None:
        clusters = DEFAULT_CONFIG.qos.clusters
    if proportions is None:
        proportions = DEFAULT_CONFIG.qos.proportions
    if len(clusters) != len(proportions):
        raise ValueError("La liste des clusters doit correspondre aux proportions.")
    total = sum(float(value) for value in proportions)
    if total <= 0:
        weights = [1.0 for _ in clusters]
    else:
        weights = [float(value) for value in proportions]
    return generator.choices(list(clusters), weights=weights, k=count)
