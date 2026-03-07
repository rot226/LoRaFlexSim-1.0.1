"""Interface CLI pour mobilesfrdth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
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




def _verbosity_level(args: argparse.Namespace) -> int:
    if getattr(args, "quiet", False):
        return 0
    return 2 if getattr(args, "verbose", False) else 1


def _print_info(args: argparse.Namespace, message: str) -> None:
    if _verbosity_level(args) >= 1:
        print(message)


def _print_verbose(args: argparse.Namespace, message: str) -> None:
    if _verbosity_level(args) >= 2:
        print(message)


def _format_eta(seconds: float) -> str:
    remaining = max(0, int(round(seconds)))
    h, rem = divmod(remaining, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
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
    start_time = time.perf_counter()
    completed = 0
    successes = 0

    def _on_run_progress(current: int, total: int, run_report: object) -> None:
        nonlocal completed, successes
        completed = current
        if getattr(run_report, "success", False):
            successes += 1
        elapsed = time.perf_counter() - start_time
        avg = elapsed / max(completed, 1)
        eta = _format_eta(avg * max(total - completed, 0))
        success_rate = (successes / max(completed, 1)) * 100.0
        _print_info(
            args,
            f"[run] {completed}/{total} | succès={success_rate:.1f}% | ETA={eta} | sortie={getattr(run_report, 'run_dir', out_dir)}",
        )

    report = orchestrator.execute_jobs(jobs, progress_callback=_on_run_progress)
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

    _print_info(args, f"{len(jobs)} jobs générés dans {output_file}")
    _print_info(args, f"Exécution terminée: {execution_summary['num_success']} succès, {execution_summary['num_failures']} échec(s)")
    _print_info(args, f"Résumé batch écrit dans {summary_file}")
    return 1 if failures else 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_order = ["metric_by_factor", "sf_distribution", "sinr_cdf"]
    phase_index: dict[str, int] = {name: idx + 1 for idx, name in enumerate(phase_order)}

    def _on_aggregate_progress(phase: str, done: int, total: int) -> None:
        if phase == "discover":
            _print_info(args, f"[aggregate] runs détectés: {done}")
            return
        if phase not in phase_index:
            return
        _print_info(args, f"[aggregate] phase={phase} ({phase_index[phase]}/{len(phase_order)}) progression={done}/{total}")

    try:
        files = aggregate_runs(inputs=args.results, output_root=out_dir, progress_callback=_on_aggregate_progress)
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
    _print_info(args, f"Agrégation écrite dans {output_file}")
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

    def _on_plot_progress(fig_name: str, out_path: Path, generated_ok: bool) -> None:
        status = "générée" if generated_ok else "ignorée"
        _print_info(args, f"[plots] {fig_name}: {status} ({out_path})")

    generated = generate_minimal_figures(
        aggregates_dir=args.aggregates_dir,
        out_dir=out_dir,
        filters=ScenarioFilters.from_tokens(args.scenario_filter),
        include_bonus=not args.no_bonus,
        progress_callback=_on_plot_progress,
    )
    report = {
        "aggregates_dir": str(args.aggregates_dir),
        "out_dir": str(out_dir),
        "num_figures": len(generated),
        "figures": [str(path) for path in generated],
    }
    output_file = out_dir / "plots_summary.json"
    _dump_json(output_file, report)
    _print_info(args, f"{len(generated)} figure(s) écrite(s) dans {out_dir}")
    _print_info(args, f"Résumé de plots écrit dans {output_file}")
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
    parser.add_argument("--verbose", action="store_true", help="Affiche plus de détails de progression.")
    parser.add_argument("--quiet", action="store_true", help="Réduit la sortie au minimum.")
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
        if getattr(args, "verbose", False) and getattr(args, "quiet", False):
            raise ValueError("--verbose et --quiet ne peuvent pas être utilisés ensemble.")
        return args.func(args)
    except ValueError as exc:
        print(f"Erreur: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
