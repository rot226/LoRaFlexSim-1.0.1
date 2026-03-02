"""Interface en ligne de commande pour mobilesfrdth."""

from __future__ import annotations

import argparse
from typing import Callable


CommandHandler = Callable[[argparse.Namespace], int]


def _run_command(_: argparse.Namespace) -> int:
    """Exécute la sous-commande `run`."""
    print("Sous-commande 'run' appelée.")
    return 0


def _aggregate_command(_: argparse.Namespace) -> int:
    """Exécute la sous-commande `aggregate`."""
    print("Sous-commande 'aggregate' appelée.")
    return 0


def _plots_command(_: argparse.Namespace) -> int:
    """Exécute la sous-commande `plots`."""
    print("Sous-commande 'plots' appelée.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI principal."""
    parser = argparse.ArgumentParser(prog="mobilesfrdth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Lancer une simulation")
    run_parser.set_defaults(handler=_run_command)

    aggregate_parser = subparsers.add_parser("aggregate", help="Agréger des résultats")
    aggregate_parser.set_defaults(handler=_aggregate_command)

    plots_parser = subparsers.add_parser("plots", help="Générer des graphiques")
    plots_parser.set_defaults(handler=_plots_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de la CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler = args.handler
    return handler(args)
