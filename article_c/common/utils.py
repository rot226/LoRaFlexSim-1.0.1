"""Utilitaires partagés pour l'article C."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI."""
    parser = argparse.ArgumentParser(description="Outils CLI pour l'article C.")
    parser.add_argument("--seed", type=int, default=None, help="Seed déterministe.")
    parser.add_argument(
        "--densities",
        type=str,
        default="0.1,0.5,1.0",
        help="Liste des densités (ex: 0.1,0.5,1.0).",
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


def parse_density_list(value: str) -> list[float]:
    """Parse une liste de densités depuis une chaîne CSV."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
