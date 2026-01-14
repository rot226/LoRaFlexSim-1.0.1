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
        "--densities",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Liste des densités (ex: 0.1 0.5 1.0).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Nombre de réplications.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Ajoute un timestamp dans les sorties.",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=snir_defaults.snir_threshold_db,
        help="Seuil SNIR/capture (dB).",
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=snir_defaults.noise_floor_dbm,
        help="Bruit thermique (dBm).",
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default=step2_defaults.traffic_mode,
        choices=("periodic", "poisson"),
        help="Modèle de trafic pour l'étape 2 (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range",
        type=float,
        default=step2_defaults.jitter_range_s,
        help="Amplitude du jitter pour l'étape 2 (secondes).",
    )
    parser.add_argument(
        "--window-duration-s",
        type=float,
        default=step2_defaults.window_duration_s,
        help="Durée d'une fenêtre de simulation (secondes).",
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


def parse_density_list(value: str | Sequence[float]) -> list[float]:
    """Parse une liste de densités depuis une chaîne CSV ou une séquence."""
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def replication_ids(count: int) -> list[int]:
    """Retourne les identifiants de réplications."""
    return list(range(1, count + 1))


def timestamp_tag(with_timezone: bool = True) -> str:
    """Retourne un timestamp ISO pour les sorties."""
    now = datetime.now(timezone.utc if with_timezone else None)
    return now.isoformat(timespec="seconds")


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
