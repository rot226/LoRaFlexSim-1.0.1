"""Orchestre une matrice d'exécutions pour l'étape 1.

Le script combine plusieurs algorithmes QoS, états SNIR, graines et
configurations de charge puis délègue l'exécution à
``scripts/run_step1_experiments.py``. Chaque exécution produit un CSV
suffixé explicitement (``_snir-on`` ou ``_snir-off``) et rangé par état
SNIR (puis par graine) dans ``results/step1/<state>/``. Ces suffixes et
le champ ``snir_state`` écrit par ``run_step1_experiments.py`` doivent
rester présents dans les CSV finaux pour permettre l'agrégation stricte
des résultats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import run_step1_experiments

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"

DEFAULT_ALGOS: Sequence[str] = ("adr", "apra", "mixra_h", "mixra_opt")
DEFAULT_SNIR_STATES: Sequence[bool] = (False, True)
DEFAULT_SEEDS: Sequence[int] = tuple(range(1, 6))
DEFAULT_NODE_COUNTS: Sequence[int] = (1000, 5000, 10000, 13000, 15000)
DEFAULT_PACKET_INTERVALS: Sequence[float] = (600.0, 300.0, 150.0)
DEFAULT_DURATION = 6 * 3600.0
STATE_LABELS = {True: "snir_on", False: "snir_off"}


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Valeur booléenne attendue (true/false, 1/0, yes/no, on/off)"
    )


def _parse_snir_window(value: str) -> str | float:
    text = str(value).strip().lower()
    if text in {"packet", "preamble", "symbol"}:
        return text
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "snir_window doit être 'packet', 'preamble', 'symbol' ou une durée en secondes."
        ) from exc


def _snir_window_label(value: str | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return f"{value:g}s"


def _snir_window_path(value: str | float | None) -> str | None:
    label = _snir_window_label(value)
    if label is None:
        return None
    return label.replace(".", "p")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=sorted(DEFAULT_ALGOS),
        default=list(DEFAULT_ALGOS),
        help="Algorithmes à tester",
    )
    parser.add_argument(
        "--with-snir",
        nargs="+",
        type=_parse_bool,
        default=None,
        metavar="BOOL",
        help=(
            "États SNIR à explorer (true/false). Utilisez explicitement "
            '"--with-snir true false" pour produire les deux jeux.'
        ),
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
        "--snir-windows",
        nargs="+",
        type=_parse_snir_window,
        default=None,
        metavar="WINDOW",
        help="Fenêtres SNIR à comparer (packet, preamble, symbol ou secondes)",
    )
    parser.add_argument(
        "--fading-std-db",
        type=float,
        default=None,
        help=(
            "Écart-type (dB) du fading aléatoire appliqué au calcul SNIR "
            "(obligatoire si SNIR activé, recommandé : 2 à 4 dB)"
        ),
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


def _snir_suffix(use_snir: bool | None) -> str:
    if use_snir is True:
        return "_snir-on"
    if use_snir is False:
        return "_snir-off"
    return ""


def _csv_filename(algorithm: str, nodes: int, packet_interval: float, use_snir: bool) -> str:
    interval = int(packet_interval) if float(packet_interval).is_integer() else packet_interval
    return f"{algorithm}_N{nodes}_T{interval}{_snir_suffix(use_snir)}.csv"


def _iter_snir_states(values: Iterable[bool] | None) -> Iterable[bool]:
    if values is None:
        return DEFAULT_SNIR_STATES
    return tuple(values)


def _run_one(
    *,
    algorithm: str,
    nodes: int,
    packet_interval: float,
    seed: int,
    use_snir: bool,
    snir_window: str | float | None,
    duration: float,
    results_dir: Path,
    skip_existing: bool,
    fading_std_db: float | None,
) -> None:
    state = STATE_LABELS.get(use_snir, "snir_unknown")
    output_dir = results_dir / state / f"seed_{seed}"
    window_label = _snir_window_path(snir_window)
    if window_label is not None:
        output_dir = output_dir / f"window_{window_label}"
    csv_path = output_dir / _csv_filename(algorithm, nodes, packet_interval, use_snir)

    if skip_existing and csv_path.exists():
        print(f"[SKIP] {csv_path} déjà présent")
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
    if fading_std_db is not None:
        argv.extend(["--fading-std-db", str(fading_std_db)])
    if use_snir:
        argv.append("--use-snir")
    else:
        argv.append("--no-snir")
    if snir_window is not None:
        argv.extend(["--snir-window", str(snir_window)])

    print(
        "[RUN] "
        f"algo={algorithm} use_snir={use_snir} seed={seed} "
        f"nodes={nodes} interval={packet_interval:g}s -> {csv_path}"
    )
    run_step1_experiments.main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        snir_states = list(_iter_snir_states(args.with_snir))
    except ValueError as exc:  # clarification immédiate pour l'utilisateur
        parser.error(str(exc))
    if any(snir_states) and args.fading_std_db is None:
        parser.error(
            "--fading-std-db est obligatoire lorsque SNIR est activé (recommandé : 2 à 4 dB)."
        )

    snir_windows: list[str | float | None]
    if args.snir_windows is None:
        snir_windows = [None]
    else:
        snir_windows = list(args.snir_windows)

    for use_snir in snir_states:
        for snir_window in snir_windows:
            for seed in args.seeds:
                for nodes in args.nodes:
                    for packet_interval in args.packet_intervals:
                        for algorithm in args.algos:
                            _run_one(
                                algorithm=algorithm,
                                nodes=nodes,
                                packet_interval=packet_interval,
                                seed=seed,
                                use_snir=use_snir,
                                snir_window=snir_window,
                                duration=args.duration,
                                results_dir=args.results_dir,
                                skip_existing=args.skip_existing,
                                fading_std_db=args.fading_std_db,
                            )


if __name__ == "__main__":
    main()
