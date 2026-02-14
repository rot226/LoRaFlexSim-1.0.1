"""Point d'entrée CLI: lancement de campagne multi-runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

from sfrd.parse.aggregate import aggregate_logs


def _load_run_single_campaign():
    """Charge le point d'entrée de simulation défini dans ``sfrd/cli.py``."""

    module_path = Path(__file__).resolve().parents[1] / "cli.py"
    spec = importlib.util.spec_from_file_location("_sfrd_core_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_campaign


def _parse_snir_modes(raw_values: str) -> list[str]:
    """Parse la liste SNIR CSV et retourne des valeurs normalisées OFF/ON."""

    mapping = {
        "off": "OFF",
        "snir_off": "OFF",
        "on": "ON",
        "snir_on": "ON",
    }
    modes: list[str] = []
    for value in raw_values.split(","):
        key = value.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError(
                f"Mode SNIR invalide: {value!r}. Attendu: OFF ou ON"
            )
        normalized = mapping[key]
        if normalized not in modes:
            modes.append(normalized)

    if not modes:
        raise argparse.ArgumentTypeError("--snir ne contient aucune valeur valide")
    return modes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance une matrice de runs SFRD (SNIR x taille x algo x réplication)."
    )
    parser.add_argument(
        "--network-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Tailles réseau, ex: --network-sizes 80 160 320 640 1280",
    )
    parser.add_argument(
        "--replications",
        type=int,
        required=True,
        help="Nombre de réplications par combinaison (>=1)",
    )
    parser.add_argument(
        "--seeds-base",
        type=int,
        required=True,
        help="Base de seed déterministe (seed = seeds_base + replication_index - 1)",
    )
    parser.add_argument(
        "--snir",
        type=_parse_snir_modes,
        default=_parse_snir_modes("OFF,ON"),
        help="Modes SNIR CSV, ex: --snir OFF,ON",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        required=True,
        help="Algorithmes, ex: --algos UCB ADR MixRA-H MixRA-Opt",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        required=True,
        help="Durée warmup en secondes",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Dossier racine des logs de campagne",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Désactive l'agrégation automatique en fin de campagne.",
    )
    args = parser.parse_args()

    if args.replications <= 0:
        parser.error("--replications doit être >= 1")
    if any(size <= 0 for size in args.network_sizes):
        parser.error("Toutes les valeurs de --network-sizes doivent être > 0")
    return args


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    run_single_campaign = _load_run_single_campaign()
    logs_root: Path = args.logs_root
    logs_root.mkdir(parents=True, exist_ok=True)

    run_index: list[dict[str, object]] = []

    for snir_mode in args.snir:
        snir_folder = f"SNIR_{snir_mode}"
        snir_cli_value = "snir_on" if snir_mode == "ON" else "snir_off"

        for network_size in args.network_sizes:
            for algorithm in args.algos:
                for replication_index in range(1, args.replications + 1):
                    seed = args.seeds_base + replication_index - 1
                    run_dir = (
                        logs_root
                        / snir_folder
                        / f"ns_{network_size}"
                        / f"algo_{algorithm}"
                        / f"seed_{seed}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_input = {
                        "snir_mode": snir_mode,
                        "network_size": int(network_size),
                        "algorithm": algorithm,
                        "replication_index": int(replication_index),
                        "seed": int(seed),
                        "warmup_s": float(args.warmup_s),
                        "seed_formula": "seed = seeds_base + replication_index - 1",
                    }
                    (run_dir / "run_input.json").write_text(
                        json.dumps(run_input, indent=2, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )

                    t0 = perf_counter()
                    result = run_single_campaign(
                        network_size=int(network_size),
                        algorithm=str(algorithm),
                        snir_mode=snir_cli_value,
                        seed=int(seed),
                        warmup_s=float(args.warmup_s),
                        output_dir=run_dir,
                    )
                    duration_s = perf_counter() - t0

                    run_status = {
                        "status": "completed",
                        "duration_s": duration_s,
                        "summary_path": str(result["summary_path"]),
                    }
                    (run_dir / "run_status.json").write_text(
                        json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )

                    run_index.append(
                        {
                            **run_input,
                            "run_dir": str(run_dir),
                            "summary_path": str(result["summary_path"]),
                        }
                    )

    (logs_root / "campaign_runs.json").write_text(
        json.dumps(run_index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_aggregate:
        aggregate_path = aggregate_logs(logs_root)
        print(f"Agrégation terminée: {aggregate_path}")
    else:
        print("Agrégation automatique ignorée (--skip-aggregate).")


if __name__ == "__main__":
    main()
