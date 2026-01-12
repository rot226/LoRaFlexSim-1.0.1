"""Lance une suite minimale Step 1 (8 exécutions light)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_step1_experiments

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_ALGOS = ("adr", "apra", "mixra_h", "mixra_opt")
DEFAULT_SNIR_STATES = (True, False)
DEFAULT_NODES = 200
DEFAULT_PACKET_INTERVAL = 300.0
DEFAULT_DURATION = 600.0
DEFAULT_SEED = 1
DEFAULT_SNIR_FADING_STD_DB = 2.0


def _snir_suffix(use_snir: bool) -> str:
    return "_snir-on" if use_snir else "_snir-off"


def _csv_name(algorithm: str, nodes: int, packet_interval: float, use_snir: bool) -> str:
    interval = int(packet_interval) if float(packet_interval).is_integer() else packet_interval
    return f"{algorithm}_N{nodes}_T{interval}{_snir_suffix(use_snir)}.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire de sortie pour les CSV générés",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=DEFAULT_NODES,
        help="Nombre de nœuds simulés (défaut : 200)",
    )
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=DEFAULT_PACKET_INTERVAL,
        help="Intervalle moyen d'émission (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Durée maximale de simulation (secondes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Graine de simulation",
    )
    parser.add_argument(
        "--snir-fading-std-db",
        type=float,
        default=DEFAULT_SNIR_FADING_STD_DB,
        help="Écart-type (dB) du fading appliqué au calcul SNIR",
    )
    return parser


def _run_experiments(
    *,
    algorithms: Iterable[str],
    snir_states: Iterable[bool],
    results_dir: Path,
    nodes: int,
    packet_interval: float,
    duration: float,
    seed: int,
    snir_fading_std_db: float,
) -> List[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Path] = []

    for algorithm in algorithms:
        for use_snir in snir_states:
            argv = [
                "--algorithm",
                algorithm,
                "--nodes",
                str(nodes),
                "--packet-interval",
                str(packet_interval),
                "--duration",
                str(duration),
                "--seed",
                str(seed),
                "--output-dir",
                str(results_dir),
                "--quiet",
            ]
            if use_snir:
                argv.extend(["--use-snir", "--fading-std-db", str(snir_fading_std_db)])
            else:
                argv.append("--no-snir")

            result = run_step1_experiments.main(argv)
            csv_path = result.get("csv_path")
            if isinstance(csv_path, Path):
                csv_paths.append(csv_path)
            else:
                csv_paths.append(results_dir / _csv_name(algorithm, nodes, packet_interval, use_snir))

    return csv_paths


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    csv_paths = _run_experiments(
        algorithms=DEFAULT_ALGOS,
        snir_states=DEFAULT_SNIR_STATES,
        results_dir=args.results_dir,
        nodes=args.nodes,
        packet_interval=args.packet_interval,
        duration=args.duration,
        seed=args.seed,
        snir_fading_std_db=args.snir_fading_std_db,
    )

    print("[SUMMARY] CSV générés :")
    for path in csv_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
