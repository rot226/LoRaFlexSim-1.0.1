"""Interface en ligne de commande pour mobilesfrdth."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import Callable, Iterator

import sfrd.cli.plot_campaign as plot_campaign
import sfrd.cli.run_campaign as run_campaign_module
import sfrd.parse.aggregate as aggregate


CommandHandler = Callable[[argparse.Namespace], int]
MIN_PYTHON = (3, 11)
MAX_PYTHON_EXCLUSIVE = (3, 13)

_GRID_PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "network_sizes": [80],
        "algos": ["ADR", "UCB"],
        "snir": "OFF,ON",
        "replications": 1,
        "seeds_base": 1,
        "warmup_s": 300.0,
    },
    "paper": {
        "network_sizes": [80, 160, 320, 640, 1280],
        "algos": ["ADR", "MixRA-H", "MixRA-Opt", "UCB"],
        "snir": "OFF,ON",
        "replications": 5,
        "seeds_base": 1,
        "warmup_s": 300.0,
    },
}


def _check_python_version() -> str | None:
    """Retourne un message d'erreur si la version Python n'est pas supportée."""
    current = sys.version_info[:2]
    if MIN_PYTHON <= current < MAX_PYTHON_EXCLUSIVE:
        return None
    return (
        "Version Python non supportée pour `mobilesfrdth`: "
        f"{current[0]}.{current[1]}. "
        "Versions supportées: >=3.11 et <3.13 (Python 3.11 recommandé)."
    )


@contextmanager
def _patched_argv(argv: list[str]) -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def _call_module_main(main_func: Callable[[], None], argv: list[str]) -> int:
    with _patched_argv(argv):
        try:
            main_func()
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                return code
            return 1
    return 0


def _run_command(args: argparse.Namespace) -> int:
    """Exécute la sous-commande `run` via `sfrd.cli.run_campaign`."""
    forwarded: list[str] = ["run_campaign"]

    uses_grid = args.grid is not None
    explicit_matrix_opts = [
        args.network_sizes is not None,
        args.algos is not None,
        args.snir is not None,
        args.replications is not None,
        args.seeds_base is not None,
        args.warmup_s is not None,
    ]
    if uses_grid and any(explicit_matrix_opts):
        print(
            "Erreur: --grid est incompatible avec --network-sizes/--algos/--snir/"
            "--replications/--seeds-base/--warmup-s. "
            "Utilisez soit le preset --grid, soit la matrice détaillée SFRD.",
            file=sys.stderr,
        )
        return 2

    if uses_grid:
        print(
            "AVERTISSEMENT: l'option --grid est une compatibilité mobilesfrdth; "
            "elle n'existe pas dans sfrd.cli.run_campaign. "
            "Le shim l'a convertie vers des options SFRD.",
            file=sys.stderr,
        )
        preset = _GRID_PRESETS[args.grid]
        forwarded.extend(["--network-sizes", *[str(v) for v in preset["network_sizes"]]])
        forwarded.extend(["--algos", *[str(v) for v in preset["algos"]]])
        forwarded.extend(["--snir", str(preset["snir"])])
        forwarded.extend(["--replications", str(preset["replications"])])
        forwarded.extend(["--seeds-base", str(preset["seeds_base"])])
        forwarded.extend(["--warmup-s", str(preset["warmup_s"])])
    else:
        if not all(explicit_matrix_opts):
            print(
                "Erreur: en mode sans --grid, fournir toutes les options SFRD: "
                "--network-sizes, --algos, --snir, --replications, --seeds-base, --warmup-s.",
                file=sys.stderr,
            )
            return 2
        forwarded.extend(["--network-sizes", *[str(v) for v in args.network_sizes]])
        forwarded.extend(["--algos", *[str(v) for v in args.algos]])
        forwarded.extend(["--snir", str(args.snir)])
        forwarded.extend(["--replications", str(args.replications)])
        forwarded.extend(["--seeds-base", str(args.seeds_base)])
        forwarded.extend(["--warmup-s", str(args.warmup_s)])

    if args.skip_algos:
        forwarded.extend(["--skip-algos", *args.skip_algos])
    if args.ucb_config is not None:
        forwarded.extend(["--ucb-config", args.ucb_config])
    if args.logs_root is not None:
        forwarded.extend(["--logs-root", args.logs_root])
    if args.allow_partial:
        forwarded.append("--allow-partial")
    if args.strict_finalize:
        forwarded.append("--strict-finalize")
    if args.skip_aggregate:
        forwarded.append("--skip-aggregate")
    if args.force_rerun:
        forwarded.append("--force-rerun")
    if args.precheck is not None:
        forwarded.extend(["--precheck", args.precheck])
    if args.keep_precheck_artifacts:
        forwarded.append("--keep-precheck-artifacts")
    if args.max_run_seconds is not None:
        forwarded.extend(["--max-run-seconds", str(args.max_run_seconds)])

    return _call_module_main(run_campaign_module.main, forwarded)


def _aggregate_command(args: argparse.Namespace) -> int:
    """Exécute la sous-commande `aggregate` via `sfrd.parse.aggregate`."""
    forwarded = ["aggregate"]
    if args.logs_root is not None:
        forwarded.extend(["--logs-root", args.logs_root])
    if args.allow_partial:
        forwarded.append("--allow-partial")
    if args.campaign_id is not None:
        forwarded.extend(["--campaign-id", args.campaign_id])
    if args.manifest is not None:
        forwarded.extend(["--manifest", args.manifest])
    if args.debug_missing:
        forwarded.append("--debug-missing")
    return _call_module_main(aggregate.main, forwarded)


def _plots_command(args: argparse.Namespace) -> int:
    """Exécute la sous-commande `plots` via `sfrd.cli.plot_campaign`."""
    forwarded = ["plot_campaign"]
    if args.campaign_id is not None:
        forwarded.extend(["--campaign-id", args.campaign_id])
    if args.logs_root is not None:
        forwarded.extend(["--logs-root", args.logs_root])
    if args.output_root is not None:
        forwarded.extend(["--output-root", args.output_root])
    if args.figures_dir is not None:
        forwarded.extend(["--figures-dir", args.figures_dir])
    if args.format is not None:
        forwarded.extend(["--format", args.format])
    return _call_module_main(plot_campaign.main, forwarded)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI principal."""
    parser = argparse.ArgumentParser(prog="mobilesfrdth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Lancer une simulation")
    run_parser.add_argument("--grid", choices=sorted(_GRID_PRESETS.keys()))
    run_parser.add_argument("--network-sizes", nargs="+", type=int)
    run_parser.add_argument("--replications", type=int)
    run_parser.add_argument("--seeds-base", type=int)
    run_parser.add_argument("--snir", type=str)
    run_parser.add_argument("--algos", nargs="+")
    run_parser.add_argument("--skip-algos", nargs="+", default=[])
    run_parser.add_argument("--warmup-s", type=float)
    run_parser.add_argument("--ucb-config", type=str)
    run_parser.add_argument("--logs-root", type=str)
    run_parser.add_argument("--allow-partial", action="store_true")
    run_parser.add_argument("--strict-finalize", action="store_true")
    run_parser.add_argument("--skip-aggregate", action="store_true")
    run_parser.add_argument("--force-rerun", action="store_true")
    run_parser.add_argument("--precheck", choices=("auto", "on", "off"), default=None)
    run_parser.add_argument("--keep-precheck-artifacts", action="store_true")
    run_parser.add_argument("--max-run-seconds", type=float)
    run_parser.set_defaults(handler=_run_command)

    aggregate_parser = subparsers.add_parser("aggregate", help="Agréger des résultats")
    aggregate_parser.add_argument("--logs-root", type=str)
    aggregate_parser.add_argument("--allow-partial", action="store_true")
    aggregate_parser.add_argument("--campaign-id", type=str)
    aggregate_parser.add_argument("--manifest", type=str)
    aggregate_parser.add_argument("--debug-missing", action="store_true")
    aggregate_parser.set_defaults(handler=_aggregate_command)

    plots_parser = subparsers.add_parser("plots", help="Générer des graphiques")
    plots_parser.add_argument("--campaign-id", type=str)
    plots_parser.add_argument("--logs-root", type=str)
    plots_parser.add_argument("--output-root", type=str)
    plots_parser.add_argument("--figures-dir", type=str)
    plots_parser.add_argument("--format", choices=["png", "svg", "pdf"], default=None)
    plots_parser.set_defaults(handler=_plots_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de la CLI."""
    version_error = _check_python_version()
    if version_error is not None:
        print(version_error, file=sys.stderr)
        return 2

    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler = args.handler
    return handler(args)
