"""Outils de génération de scénarios et de jobs pour mobilesfrdth."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
from typing import Any

GRID_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_GRID_KEYS = {"N", "mode", "algo", "reps", "seed_base"}
ALLOWED_GRID_KEYS = REQUIRED_GRID_KEYS | {
    "model",
    "speed",
    "gateways",
    "sigma",
    "sigma_shadowing",
    "duration_s",
    "period_s",
    "payload_size",
    "time_bin_s",
}

MODEL_ALIASES = {
    "RWP": "RWP",
    "SMOOTH": "SMOOTH",
}

MODE_ALIASES = {
    "SNIR_OFF": "SNIR_OFF",
    "SNIR_ON": "SNIR_ON",
    "OFF": "SNIR_OFF",
    "ON": "SNIR_ON",
}

ALGO_ALIASES = {
    "ADR": "ADR",
    "ADR_MIXRA": "ADR_MIXRA",
    "UCB": "UCB",
    "UCB_FORGET": "UCB_FORGET",
}


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


def _normalize_enum(value: Any, *, key: str, aliases: dict[str, str]) -> str:
    raw = str(value).strip()
    if raw == "":
        raise ValueError(f"La clé '{key}' contient une valeur vide.")
    canonical = raw.upper().replace("-", "_")
    if canonical not in aliases:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Valeur invalide pour '{key}': '{raw}'. Valeurs autorisées: {allowed}.")
    return aliases[canonical]


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
        if key == "":
            raise ValueError(f"Entrée de grille invalide '{chunk}': clé vide.")
        if not GRID_KEY_PATTERN.match(key):
            raise ValueError(
                f"Nom de clé invalide '{key}'. Utiliser [A-Za-z_][A-Za-z0-9_]*."
            )

        if key not in ALLOWED_GRID_KEYS:
            allowed = ", ".join(sorted(ALLOWED_GRID_KEYS))
            raise ValueError(f"Clé inconnue '{key}'. Clés autorisées: {allowed}.")

        raw_items = [item.strip() for item in raw_values.split(",")]
        if any(item == "" for item in raw_items):
            raise ValueError(f"La clé '{key}' contient une valeur vide.")

        values = [_parse_scalar(v) for v in raw_items]
        if not values:
            raise ValueError(f"La clé '{key}' n'a aucune valeur.")

        if key == "model":
            values = [_normalize_enum(value, key=key, aliases=MODEL_ALIASES) for value in values]
        elif key == "mode":
            values = [_normalize_enum(value, key=key, aliases=MODE_ALIASES) for value in values]
        elif key == "algo":
            values = [_normalize_enum(value, key=key, aliases=ALGO_ALIASES) for value in values]

        result[key] = values

    missing = sorted(REQUIRED_GRID_KEYS - set(result))
    if missing:
        raise ValueError(f"Clés obligatoires manquantes: {', '.join(missing)}.")

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

    if "seed_base" in grid:
        for seed_base in grid["seed_base"]:
            if not isinstance(seed_base, int) or not (checks.min_seed <= seed_base <= checks.max_seed):
                raise ValueError(f"Toutes les valeurs seed_base doivent être dans [{checks.min_seed}, {checks.max_seed}].")

    if "gateways" in grid:
        for gateways in grid["gateways"]:
            if not isinstance(gateways, int) or gateways < 1:
                raise ValueError("Toutes les valeurs gateways doivent être des entiers >= 1.")

    if "sigma" in grid:
        for sigma in grid["sigma"]:
            if not isinstance(sigma, (int, float)) or sigma < 0:
                raise ValueError("Toutes les valeurs sigma doivent être numériques et >= 0.")

    if "sigma_shadowing" in grid:
        for sigma in grid["sigma_shadowing"]:
            if not isinstance(sigma, (int, float)) or sigma < 0:
                raise ValueError("Toutes les valeurs sigma_shadowing doivent être numériques et >= 0.")


def _build_run_id(params: dict[str, Any], rep: int, seed: int) -> str:
    model = str(params.get("model", "RWP")).lower()
    mode = str(params["mode"]).lower()
    algo = str(params["algo"]).lower()
    n_val = params["N"]
    return f"n{n_val}_model-{model}_mode-{mode}_algo-{algo}_rep-{rep:03d}_seed-{seed}"


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

    keys = [key for key in grid if key not in {"reps", "seed_base"}]
    combinations = list(product(*(grid[k] for k in keys)))
    jobs: list[dict[str, Any]] = []

    job_index = 1
    for values in combinations:
        base_params = dict(zip(keys, values, strict=True))
        reps_count = reps if reps is not None else int(grid["reps"][0])
        seed_origin = seed if seed is not None else int(grid["seed_base"][0])

        for rep in range(1, reps_count + 1):
            rep_seed = seed_origin + rep - 1
            params = dict(base_params)
            params["rep"] = rep
            params["seed"] = rep_seed
            params["run_id"] = _build_run_id(params, rep, rep_seed)
            if sf_range is not None:
                params.setdefault("sf_min", sf_range[0])
                params.setdefault("sf_max", sf_range[1])

            jobs.append(
                {
                    "job_id": f"job_{job_index:04d}",
                    "config": str(config_path),
                    "output": str(output_root / params["run_id"]),
                    "params": params,
                }
            )
            job_index += 1
    return jobs
