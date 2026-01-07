"""Génère une baseline Step 1 (summary.csv + fichiers SNIR dédiés)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.aggregate_step1_results import aggregate_step1_results
from scripts.run_step1_matrix import main as run_step1_matrix

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_ALGOS: Sequence[str] = ("mixra_opt",)
DEFAULT_SNIR_STATES: Sequence[bool] = (False, True)
DEFAULT_SEEDS: Sequence[int] = (1, 2, 3)
DEFAULT_NODE_COUNTS: Sequence[int] = (1000, 5000)
DEFAULT_PACKET_INTERVALS: Sequence[float] = (300.0, 600.0)
DEFAULT_DURATION = 6 * 3600.0

EXPECTED_BASELINE_CSVS = (
    "summary.csv",
    "raw_index.csv",
    "summary_snir_on.csv",
    "summary_snir_off.csv",
    "raw_index_snir_on.csv",
    "raw_index_snir_off.csv",
)


def expected_baseline_csvs(results_dir: Path) -> List[Path]:
    return [results_dir / name for name in EXPECTED_BASELINE_CSVS]


def missing_baseline_csvs(results_dir: Path) -> List[Path]:
    return [path for path in expected_baseline_csvs(results_dir) if not path.exists()]


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Valeur booléenne attendue (true/false, 1/0, yes/no, on/off)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(DEFAULT_ALGOS),
        help="Algorithmes à inclure dans la baseline (défaut : mixra_opt)",
    )
    parser.add_argument(
        "--with-snir",
        nargs="+",
        type=_parse_bool,
        default=list(DEFAULT_SNIR_STATES),
        metavar="BOOL",
        help='États SNIR à inclure (ex: "--with-snir true false").',
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Graines utilisées pour la simulation",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=list(DEFAULT_NODE_COUNTS),
        help="Charges en nombre de nœuds",
    )
    parser.add_argument(
        "--packet-intervals",
        nargs="+",
        type=float,
        default=list(DEFAULT_PACKET_INTERVALS),
        help="Périodes moyennes d'émission (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Durée maximale des simulations (secondes)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire racine pour les CSV générés",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ignore les combinaisons dont le CSV de sortie existe déjà",
    )
    return parser


def _iter_bool_args(values: Iterable[bool]) -> List[str]:
    return ["true" if value else "false" for value in values]


def _run_matrix(args: argparse.Namespace) -> None:
    argv: List[str] = [
        "--algos",
        *[str(algo) for algo in args.algos],
        "--with-snir",
        *_iter_bool_args(args.with_snir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--nodes",
        *[str(nodes) for nodes in args.nodes],
        "--packet-intervals",
        *[str(interval) for interval in args.packet_intervals],
        "--duration",
        str(args.duration),
        "--results-dir",
        str(args.results_dir),
    ]
    if args.skip_existing:
        argv.append("--skip-existing")
    run_step1_matrix(argv)


def _validate_outputs(results_dir: Path) -> None:
    missing = missing_baseline_csvs(results_dir)
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "CSV baseline manquants après l'agrégation : " f"{missing_list}"
        )


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("[RUN] Exécution de la matrice Step 1 baseline …")
    _run_matrix(args)

    print("[RUN] Agrégation des CSV Step 1 …")
    aggregate_step1_results(
        args.results_dir, strict_snir_detection=True, split_snir=True
    )

    _validate_outputs(args.results_dir)
    print("[OK] Baseline Step 1 générée :")
    for path in expected_baseline_csvs(args.results_dir):
        print(f" - {path}")


if __name__ == "__main__":
    main()
