"""Exemple basique de simulation LoRaFlexSim"""

import os
import sys
import argparse

# Ajoute le répertoire parent pour pouvoir importer le package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple basique de simulation LoRaFlexSim")
    parser.add_argument("--nodes", type=int, default=20, help="Nombre de nœuds")
    parser.add_argument("--steps", type=int, default=500, help="Durée de la simulation")
    parser.add_argument(
        "--dump-intervals",
        action="store_true",
        help="Exporte les intervalles dans des fichiers Parquet",
    )
    args = parser.parse_args()

    sim = Simulator(
        num_nodes=args.nodes,
        packet_interval=10.0,
        transmission_mode="Random",
        adr_method="avg",
        dump_intervals=args.dump_intervals,
    )
    sim.run(args.steps)
    print(sim.get_metrics())
