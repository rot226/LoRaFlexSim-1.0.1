"""Exécution d'un scénario FLoRa prêt à l'emploi."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

# Ajoute le répertoire parent pour résoudre les imports du package lorsqu'il est exécuté
# directement (``python examples/run_flora_example.py``).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.adr_standard_1 import apply as adr1


# Module-level reference to the most recently executed simulation.
#
# Les tests chargent ce module via :func:`runpy.run_path` et s'attendent à
# retrouver un objet ``sim`` dans l'espace global, ce qui n'était pas le cas
# précédemment.  Nous initialisons donc ce nom avec ``None`` et le mettons à
# jour lors de l'exécution.
sim: Simulator | None = None

CONFIG = "flora-master/simulations/examples/n100-gw1.ini"


def run_example(*, steps: int = 1000, quiet: bool = False) -> Mapping[str, Any]:
    """Exécute le scénario FLoRa de référence et retourne les métriques."""

    sim_instance = Simulator(
        flora_mode=True,
        config_file=CONFIG,
        seed=1,
        adr_method="avg",
    )
    adr1(sim_instance, degrade_channel=True, profile="flora", capture_mode="flora")
    sim_instance.run(steps)
    metrics = sim_instance.get_metrics()
    # Expose the simulation instance for introspection by les tests.
    global sim
    sim = sim_instance
    if not quiet:
        print(metrics)
    return metrics


def main() -> Mapping[str, Any]:
    """Point d'entrée CLI : exécute le scénario et affiche les métriques."""

    return run_example(quiet=False)


if __name__ == "__main__":
    main()
