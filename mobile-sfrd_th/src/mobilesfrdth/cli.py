"""Interface CLI pour mobilesfrdth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .scenarios import generate_jobs, parse_grid_spec
from .plotting.plots import ScenarioFilters, generate_minimal_figures, validate_aggregates_inputs
from .simulator.engine import GridRunOrchestrator
from .simulator.io import aggregate_runs


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Fichier introuvable: {path}")
    return path


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Chemin introuvable: {path}")
    return path


def _sf_range(value: str) -> tuple[int, int]:
    token = value.strip()
    sep = "-" if "-" in token else ":" if ":" in token else None
    if sep is None:
        raise argparse.ArgumentTypeError("Format attendu pour --sf-range: min-max (ex: 7-12).")
    left, right = [part.strip() for part in token.split(sep, 1)]
    if not left or not right:
        raise argparse.ArgumentTypeError("--sf-range incomplet, utiliser min-max.")
    try:
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sf-range doit contenir des entiers.") from exc


def _positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} doit être un entier.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{name} doit être >= 1.")
    return parsed


def _seed_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed doit être un entier.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("--seed doit être >= 0.")
    return parsed


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_job_payloads(results: Iterable[Path]) -> list[dict]:
    payloads: list[dict] = []
    for result in results:
        if result.is_dir():
            candidate = result / "jobs.json"
            if not candidate.is_file():
                raise ValueError(f"Répertoire résultat sans jobs.json: {result}")
            target = candidate
        else:
            target = result
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                payloads.append(data)
            else:
                raise ValueError(f"Format JSON inattendu dans {target} (objet requis).")
    return payloads


def cmd_run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        grid = parse_grid_spec(args.grid)
        jobs = generate_jobs(
            config_path=args.config,
            output_root=out_dir,
            grid=grid,
            seed=args.seed,
            reps=args.reps,
            sf_range=args.sf_range,
        )
    except ValueError as exc:
        raise SystemExit(f"Erreur de validation: {exc}") from exc

    payload = {
        "config": str(args.config),
        "grid": grid,
        "seed": args.seed,
        "reps": args.reps,
        "sf_range": list(args.sf_range) if args.sf_range else None,
        "jobs": jobs,
        "num_jobs": len(jobs),
    }
    output_file = out_dir / "jobs.json"
    _dump_json(output_file, payload)

    orchestrator = GridRunOrchestrator(output_root=out_dir)
    report = orchestrator.execute_jobs(jobs)
    failures = [
        {"run_id": item.run_id, "error": item.error, "run_dir": str(item.run_dir)}
        for item in report.failed_reports
    ]
    execution_summary = {
        "num_jobs": len(jobs),
        "num_success": len(jobs) - len(failures),
        "num_failures": len(failures),
        "failures": failures,
    }
    summary_file = out_dir / "batch_summary.json"
    _dump_json(summary_file, execution_summary)

    print(f"{len(jobs)} jobs générés dans {output_file}")
    print(f"Exécution terminée: {execution_summary['num_success']} succès, {execution_summary['num_failures']} échec(s)")
    print(f"Résumé batch écrit dans {summary_file}")
    return 1 if failures else 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = aggregate_runs(inputs=args.results, output_root=out_dir)
    except (ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"Erreur pendant l'agrégation: {exc}")
        return 2

    manifest = {
        "num_inputs": len(args.results),
        "sources": [str(path) for path in args.results],
        "files": {name: str(path) for name, path in files.items()},
    }
    output_file = out_dir / "aggregate.json"
    _dump_json(output_file, manifest)
    print(f"Agrégation écrite dans {output_file}")
    return 0


def cmd_plots(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = validate_aggregates_inputs(args.aggregates_dir)
    if errors:
        print("Prérequis manquants pour plotting:")
        for err in errors:
            print(f"- {err}")
        return 2

    generated = generate_minimal_figures(
        aggregates_dir=args.aggregates_dir,
        out_dir=out_dir,
        filters=ScenarioFilters.from_tokens(args.scenario_filter),
        include_bonus=not args.no_bonus,
    )
    report = {
        "aggregates_dir": str(args.aggregates_dir),
        "out_dir": str(out_dir),
        "num_figures": len(generated),
        "figures": [str(path) for path in generated],
    }
    output_file = out_dir / "plots_summary.json"
    _dump_json(output_file, report)
    print(f"{len(generated)} figure(s) écrite(s) dans {out_dir}")
    print(f"Résumé de plots écrit dans {output_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobilesfrdth",
        description="CLI de campagnes mobile-sfrd_th: génération, agrégation et préparation des plots.",
        epilog=(
            "Exemple grille: N=50,100,160;speed=1,3;seed=1,2\n"
            "Exemple run: mobilesfrdth run --config experiments/default.yaml --out runs --grid 'N=50,100;speed=1,3'"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Génère les jobs puis exécute la campagne.")
    run_parser.add_argument("--config", required=True, type=_existing_file, help="Fichier de configuration de base.")
    run_parser.add_argument("--out", required=True, type=Path, help="Répertoire de sortie (jobs.json, results/<run_id>/...).")
    run_parser.add_argument(
        "--grid",
        required=True,
        help="Grille de sweep au format clé=v1,v2;clé2=v3,v4 (ex: N=50,100;speed=1,3).",
    )
    run_parser.add_argument(
        "--seed",
        type=_seed_int,
        default=None,
        help="Seed globale (entier >= 0) injectée dans chaque job si absente de --grid.",
    )
    run_parser.add_argument(
        "--reps",
        type=lambda value: _positive_int(value, name="--reps"),
        default=None,
        help="Nombre de répétitions par job (entier >= 1).",
    )
    run_parser.add_argument(
        "--sf-range",
        type=_sf_range,
        default=None,
        help="Plage SF globale, format min-max (bornes attendues: 7-12).",
    )
    run_parser.set_defaults(func=cmd_run)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Agrège plusieurs runs et produit les CSV standards dans aggregates/."
    )
    aggregate_parser.add_argument(
        "--results",
        required=True,
        nargs="+",
        type=_existing_path,
        help="Un ou plusieurs chemins vers des runs (ou un dossier contenant results/<run_id>/...).",
    )
    aggregate_parser.add_argument("--out", required=True, type=Path, help="Répertoire où écrire aggregates/*.csv et aggregate.json.")
    aggregate_parser.set_defaults(func=cmd_aggregate)

    plots_parser = subparsers.add_parser(
        "plots", help="Génère les figures fig01..fig10 (et bonus fig11..fig12) depuis aggregates/*.csv."
    )
    plots_parser.add_argument(
        "--aggregates-dir",
        required=True,
        type=_existing_path,
        help="Répertoire contenant les CSV d'agrégats (metric_by_factor.csv, sinr_cdf.csv, ...).",
    )
    plots_parser.add_argument("--out", required=True, type=Path, help="Répertoire où écrire les figures PNG.")
    plots_parser.add_argument(
        "--scenario-filter",
        action="append",
        default=[],
        help="Filtre clé=val1,val2 (répétable), ex: --scenario-filter algo=ucb --scenario-filter mobility_model=rwp.",
    )
    plots_parser.add_argument("--no-bonus", action="store_true", help="Désactive les figures bonus fig11/fig12.")
    plots_parser.set_defaults(func=cmd_plots)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except ValueError as exc:
        print(f"Erreur: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
