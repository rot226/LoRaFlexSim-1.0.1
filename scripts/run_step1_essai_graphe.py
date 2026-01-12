"""Lance une campagne Step 1 pour l'essai graphe.

Paramètres fixés (<=1000 nœuds) :
- algos = adr, apra, mixra_h, mixra_opt
- snir = on/off
- seeds = 1,2
- nodes = 300, 600, 1000
- packet_interval = 300s et 600s
- duration = 1800s

Sortie : essai_graphe/step1/<algo>/ avec suffixes _snir-on.csv et _snir-off.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import run_step1_experiments

DEFAULT_RESULTS_DIR = ROOT_DIR / "essai_graphe" / "step1"
ALGORITHMS = ("adr", "apra", "mixra_h", "mixra_opt")
SNIR_STATES = (False, True)
SEEDS = (1, 2)
NODE_COUNTS = (300, 600, 1000)
PACKET_INTERVALS = (300.0, 600.0)
DURATION = 1800.0


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
    parser.add_argument(
        "--snir",
        nargs="+",
        type=_parse_bool,
        default=list(SNIR_STATES),
        metavar="BOOL",
        help='États SNIR à explorer (ex: "--snir true false").',
    )
    return parser


def _snir_suffix_from_path(path: Path) -> str:
    name = path.name
    if "_snir-on" in name:
        return "_snir-on"
    if "_snir-off" in name:
        return "_snir-off"
    return ""


def _snir_suffix(use_snir: bool) -> str:
    return "_snir-on" if use_snir else "_snir-off"


def _interval_label(packet_interval: float) -> str:
    return str(int(packet_interval)) if float(packet_interval).is_integer() else str(packet_interval)


def _run_one(
    *,
    algorithm: str,
    nodes: int,
    packet_interval: float,
    seed: int,
    use_snir: bool,
    duration: float,
    results_dir: Path,
    skip_existing: bool,
) -> None:
    output_dir = results_dir / algorithm / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    interval_label = _interval_label(packet_interval)
    final_name = (
        f"{algorithm}_N{nodes}_T{interval_label}_seed{seed}{_snir_suffix(use_snir)}.csv"
    )
    final_path = results_dir / algorithm / final_name
    if skip_existing and final_path.exists():
        print(f"[SKIP] {final_path} déjà présent")
        return

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
        str(output_dir),
        "--quiet",
    ]
    if use_snir:
        argv.append("--use-snir")
    else:
        argv.append("--no-snir")

    result = run_step1_experiments.main(argv)
    csv_path = result["csv_path"]

    suffix = _snir_suffix_from_path(csv_path)
    if suffix and suffix not in final_name:
        final_path = results_dir / algorithm / f"{algorithm}_N{nodes}_T{interval_label}_seed{seed}{suffix}.csv"

    final_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.replace(final_path)

    if not any(output_dir.iterdir()):
        output_dir.rmdir()

    print(f"[OK] {final_path}")


def _iter_snir_states(values: Iterable[bool]) -> tuple[bool, ...]:
    return tuple(values) if values else SNIR_STATES


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    for use_snir in _iter_snir_states(args.snir):
        for seed in SEEDS:
            for nodes in NODE_COUNTS:
                for packet_interval in PACKET_INTERVALS:
                    for algorithm in ALGORITHMS:
                        _run_one(
                            algorithm=algorithm,
                            nodes=nodes,
                            packet_interval=packet_interval,
                            seed=seed,
                            use_snir=use_snir,
                            duration=DURATION,
                            results_dir=args.results_dir,
                            skip_existing=args.skip_existing,
                        )


if __name__ == "__main__":
    main()
