"""Outils de génération de scénarios et de jobs pour mobilesfrdth."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
from typing import Any

GRID_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class JobValidationConfig:
    """Contraintes de validation appliquées aux paramètres de campagne."""

    min_sf: int = 7
    max_sf: int = 12
    min_seed: int = 0
    max_seed: int = 2**32 - 1


DEFAULT_VALIDATION = JobValidationConfig()


def _parse_scalar(value: str) -> Any:
    token = value.strip()
    if token == "":
        raise ValueError("Valeur vide rencontrée dans la grille.")
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_grid_spec(grid_spec: str) -> dict[str, list[Any]]:
    """Parse une grille au format ``cle=v1,v2;autre=...``.

    Exemples
    --------
    ``N=50,100,160;speed=1,3``
    """

    spec = (grid_spec or "").strip()
    if not spec:
        raise ValueError("--grid ne peut pas être vide.")

    result: dict[str, list[Any]] = {}
    chunks = [chunk.strip() for chunk in spec.split(";") if chunk.strip()]
    if not chunks:
        raise ValueError("--grid doit contenir au moins un couple clé=liste.")

    for chunk in chunks:
        if "=" not in chunk:
            raise ValueError(f"Entrée de grille invalide '{chunk}': '=' manquant.")
        key, raw_values = chunk.split("=", 1)
        key = key.strip()
        if not GRID_KEY_PATTERN.match(key):
            raise ValueError(
                f"Nom de clé invalide '{key}'. Utiliser [A-Za-z_][A-Za-z0-9_]*."
            )

        values = [_parse_scalar(v) for v in raw_values.split(",")]
        if not values:
            raise ValueError(f"La clé '{key}' n'a aucune valeur.")
        result[key] = values

    return result


def _validate_grid_values(grid: dict[str, list[Any]], checks: JobValidationConfig) -> None:
    if "N" in grid:
        for n in grid["N"]:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("Toutes les valeurs N doivent être des entiers strictement positifs.")

    if "speed" in grid:
        for speed in grid["speed"]:
            if not isinstance(speed, (int, float)) or speed < 0:
                raise ValueError("Toutes les vitesses doivent être numériques et >= 0.")

    if "sf" in grid:
        for sf in grid["sf"]:
            if not isinstance(sf, int) or not (checks.min_sf <= sf <= checks.max_sf):
                raise ValueError(
                    f"Toutes les valeurs sf doivent être des entiers dans [{checks.min_sf}, {checks.max_sf}]."
                )

    if "seed" in grid:
        for seed in grid["seed"]:
            if not isinstance(seed, int) or not (checks.min_seed <= seed <= checks.max_seed):
                raise ValueError(f"Toutes les seeds doivent être dans [{checks.min_seed}, {checks.max_seed}].")

    if "reps" in grid:
        for reps in grid["reps"]:
            if not isinstance(reps, int) or reps < 1:
                raise ValueError("Toutes les valeurs reps doivent être des entiers >= 1.")


def validate_run_parameters(
    *,
    seed: int | None,
    reps: int | None,
    sf_range: tuple[int, int] | None,
    checks: JobValidationConfig = DEFAULT_VALIDATION,
) -> None:
    """Valide les paramètres globaux de la commande ``run``."""

    if seed is not None and not (checks.min_seed <= seed <= checks.max_seed):
        raise ValueError(f"--seed doit être dans [{checks.min_seed}, {checks.max_seed}].")
    if reps is not None and reps < 1:
        raise ValueError("--reps doit être >= 1.")
    if sf_range is not None:
        sf_min, sf_max = sf_range
        if sf_min > sf_max:
            raise ValueError("--sf-range invalide: borne min > borne max.")
        if sf_min < checks.min_sf or sf_max > checks.max_sf:
            raise ValueError(f"--sf-range doit rester dans [{checks.min_sf}, {checks.max_sf}].")


def generate_jobs(
    *,
    config_path: Path,
    output_root: Path,
    grid: dict[str, list[Any]],
    seed: int | None = None,
    reps: int | None = None,
    sf_range: tuple[int, int] | None = None,
    checks: JobValidationConfig = DEFAULT_VALIDATION,
) -> list[dict[str, Any]]:
    """Génère la liste des jobs (produit cartésien) à partir de la grille."""

    validate_run_parameters(seed=seed, reps=reps, sf_range=sf_range, checks=checks)
    _validate_grid_values(grid, checks)

    keys = list(grid.keys())
    combinations = list(product(*(grid[k] for k in keys)))
    jobs: list[dict[str, Any]] = []

    for index, values in enumerate(combinations, start=1):
        params = dict(zip(keys, values, strict=True))
        if seed is not None:
            params.setdefault("seed", seed)
        if reps is not None:
            params.setdefault("reps", reps)
        if sf_range is not None:
            params.setdefault("sf_min", sf_range[0])
            params.setdefault("sf_max", sf_range[1])

        jobs.append(
            {
                "job_id": f"job_{index:04d}",
                "config": str(config_path),
                "output": str(output_root / f"job_{index:04d}"),
                "params": params,
            }
        )
    return jobs
