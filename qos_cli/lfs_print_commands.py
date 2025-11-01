"""Outil CLI pour générer les commandes LoRaFlexSim à exécuter.

Ce script lit un fichier ``scenarios.yaml`` et imprime sur ``stdout`` les
commandes permettant de lancer ``lora_flex_sim.py`` pour chaque scénario et
méthode pris en charge. Aucune commande n'est exécutée : la sortie peut être
redirigée dans un script shell ou inspectée directement par l'utilisateur.

Exemple d'utilisation::

    python qos_cli/lfs_print_commands.py --config qos_cli/scenarios.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("scenarios.yaml")
SUPPORTED_METHODS = ("ADR", "APRA_like", "MixRA_H", "MixRA_Opt", "Greedy")


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Imprime les commandes lora_flex_sim.py à lancer pour chaque scénario "
            "et méthode spécifiés."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Chemin du fichier scenarios.yaml à charger (défaut: %(default)s).",
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="methods",
        action="append",
        choices=SUPPORTED_METHODS,
        help=(
            "Méthode(s) à inclure. Peut être spécifié plusieurs fois. "
            "Défaut: toutes les méthodes prises en charge."
        ),
    )
    parser.add_argument(
        "-s",
        "--scenario",
        dest="scenarios",
        action="append",
        help=(
            "Nom(s) de scénario à inclure. Peut être spécifié plusieurs fois. "
            "Défaut: tous les scénarios présents dans le YAML."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "N'imprimer que des indications textuelles. Utile pour le débogage "
            "ou les tests d'intégration."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher des informations supplémentaires sur la configuration lue.",
    )
    return parser


def load_scenarios(path: Path) -> dict:
    """Charge le fichier YAML et retourne les scénarios déclarés."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    if not isinstance(data, dict):
        raise ValueError("Le fichier YAML doit contenir un dictionnaire à la racine.")

    scenarios = data.get("scenarios", {})
    if not isinstance(scenarios, dict):
        raise ValueError("La clé 'scenarios' doit contenir un dictionnaire de cas d'usage.")

    return scenarios


def format_commands(
    scenarios: dict,
    methods: Iterable[str],
) -> List[str]:
    """Retourne la liste des commandes à exécuter pour chaque scénario et méthode."""
    commands: List[str] = []
    for scenario_name in scenarios:
        for method in methods:
            commands.append(
                "python lora_flex_sim.py --scenario {scenario} --method {method} "
                "--out results/{method}/{scenario}/".format(
                    scenario=scenario_name,
                    method=method,
                )
            )
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.methods:
        methods = tuple(dict.fromkeys(args.methods))
    else:
        methods = SUPPORTED_METHODS

    try:
        scenarios = load_scenarios(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:  # pragma: no cover - CLI simple
        parser.error(str(exc))
        return 2  # unreachable, mais garde un code de retour explicite.

    scenario_names = list(scenarios.keys())
    if args.scenarios:
        wanted = set(args.scenarios)
        scenario_names = [name for name in scenario_names if name in wanted]

    if args.verbose:
        print(f"# Chargement du fichier {args.config}")
        print(f"# Méthodes retenues: {', '.join(methods)}")
        if args.scenarios:
            print(f"# Scénarios filtrés: {', '.join(scenario_names)}")
        else:
            print(f"# Scénarios disponibles: {', '.join(scenario_names)}")

    commands = format_commands({name: scenarios[name] for name in scenario_names}, methods)

    if args.dry_run:
        print("# Mode dry-run: les commandes suivantes ne sont pas exécutées.")

    for command in commands:
        print(command)

    if not commands and args.verbose:
        print("# Aucun scénario/méthode à afficher.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
