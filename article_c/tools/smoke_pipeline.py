"""Pipeline reproductible (clean -> run_all -> make_all_plots -> verify_all).

Objectif:
- enchaîner automatiquement les 4 étapes contractuelles ;
- échouer immédiatement au premier écart (code retour non nul) ;
- écrire un rapport final JSON avec le détail des étapes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SIZES = (80, 160, 320, 640, 1280)
DEFAULT_REPORT_PATH = ROOT_DIR / "article_c" / "smoke_pipeline_report.json"


@dataclass
class StepReport:
    name: str
    command: list[str]
    started_at: float
    ended_at: float
    duration_s: float
    returncode: int
    status: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute un pipeline reproductible: clean, run_all, make_all_plots, verify_all."
        )
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Tailles réseau passées à run_all et make_all_plots.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Nombre de réplications passées à run_all et verify_all.",
    )
    parser.add_argument(
        "--seeds-base",
        type=int,
        default=1,
        help="Graine de base passée à run_all.",
    )
    parser.add_argument(
        "--log-level",
        choices=("quiet", "info", "debug"),
        default="info",
        help="Niveau de log pour run_all.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Fichier JSON du rapport final.",
    )
    return parser


def _run_step(name: str, command: list[str]) -> StepReport:
    print(f"\n=== [{name}] ===")
    print("$", " ".join(command))
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    ended = time.time()
    status = "ok" if completed.returncode == 0 else "failed"
    print(f"[{name}] status={status} returncode={completed.returncode} duration={ended - started:.2f}s")
    return StepReport(
        name=name,
        command=command,
        started_at=started,
        ended_at=ended,
        duration_s=ended - started,
        returncode=completed.returncode,
        status=status,
    )


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    python = sys.executable
    sizes = [str(value) for value in args.network_sizes]

    steps: list[tuple[str, list[str]]] = [
        (
            "clean",
            [
                python,
                "-m",
                "article_c.run_all",
                "--allow-non-article-c",
                "--clean-hard",
                "--skip-step1",
                "--skip-step2",
                "--log-level",
                args.log_level,
            ],
        ),
        (
            "run_all",
            [
                python,
                "-m",
                "article_c.run_all",
                "--allow-non-article-c",
                "--network-sizes",
                *sizes,
                "--replications",
                str(args.replications),
                "--seeds_base",
                str(args.seeds_base),
                "--log-level",
                args.log_level,
            ],
        ),
        (
            "make_all_plots",
            [
                python,
                "-m",
                "article_c.make_all_plots",
                "--network-sizes",
                *sizes,
            ],
        ),
        (
            "verify_all",
            [
                python,
                "-m",
                "article_c.tools.verify_all",
                "--replications",
                str(args.replications),
            ],
        ),
    ]

    pipeline_started = time.time()
    executed_reports: list[StepReport] = []
    failure_reason = ""

    for name, command in steps:
        report = _run_step(name, command)
        executed_reports.append(report)
        if report.returncode != 0:
            failure_reason = (
                f"Écart contractuel détecté à l'étape '{name}' "
                f"(code retour {report.returncode})."
            )
            break

    pipeline_ended = time.time()
    success = not failure_reason

    final_payload: dict[str, object] = {
        "success": success,
        "failure_reason": failure_reason,
        "network_sizes": args.network_sizes,
        "replications": args.replications,
        "seeds_base": args.seeds_base,
        "pipeline_started_at": pipeline_started,
        "pipeline_ended_at": pipeline_ended,
        "pipeline_duration_s": pipeline_ended - pipeline_started,
        "executed_steps": [asdict(item) for item in executed_reports],
        "skipped_steps": [name for name, _ in steps[len(executed_reports):]],
    }
    _write_report(args.report_path, final_payload)

    print("\n=== RAPPORT FINAL ===")
    print(f"- success: {success}")
    if failure_reason:
        print(f"- failure_reason: {failure_reason}")
    print(f"- report_path: {args.report_path}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
