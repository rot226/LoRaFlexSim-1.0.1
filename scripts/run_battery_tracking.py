"""Run a small simulation and track battery levels after each event.

This script executes the simulator step by step, collecting the remaining
energy of each node after every processed event.  The collected data is stored
in ``results/battery_tracking.csv`` with columns ``time``, ``node_id``,
``energy_j``, ``capacity_j``, ``alive`` and ``replicate``.  Multiple replicates
can be executed to gather statistics across runs.

Usage::

    python scripts/run_battery_tracking.py --nodes 5 --packets 3 --seed 1 --replicates 2

Le trafic est périodique (intervalle de 1 seconde), sans mobilité et avec
une seule passerelle par défaut afin de garantir des mesures reproductibles.
Les options ``--transmission-mode``, ``--packet-interval``, ``--mobility`` et
``--gateways`` permettent de personnaliser ces paramètres si nécessaire.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Iterable

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import Simulator  # noqa: E402

try:  # pandas is optional but required for CSV export
    import pandas as pd
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"pandas is required for this script: {exc}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# Default battery capacity in joules for each node.  A finite value is required
# to observe the remaining energy decreasing over time.
DEFAULT_BATTERY_J = 1000.0


def _collect(sim: Simulator, replicate: int) -> Iterable[dict[str, float | int | bool]]:
    """Yield a record for each node with current time and remaining energy."""
    for node in sim.nodes:
        # Prefer explicit battery attribute when available
        energy = getattr(node, "battery_remaining_j", None)
        if energy is None:
            # Fallback to generic energy attributes if present
            energy = getattr(node, "remaining_energy", None)
        if energy is None:
            energy = getattr(node, "energy_total", None)
        if energy is None:
            energy = getattr(node, "energy_consumed", 0.0)
        capacity = getattr(node, "battery_capacity_j", DEFAULT_BATTERY_J)
        yield {
            "time": sim.current_time,
            "node_id": node.id,
            "energy_j": energy,
            "capacity_j": capacity,
            "replicate": replicate,
            "alive": getattr(node, "alive", True),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track node battery energy")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument(
        "--packets", type=int, default=3, help="Packets to send per node"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of simulation replicates",
    )
    parser.add_argument(
        "--gateways",
        type=int,
        default=1,
        help="Nombre de passerelles dans la simulation",
    )
    parser.add_argument(
        "--transmission-mode",
        default="Periodic",
        help="Mode de transmission (ex. Periodic, Random)",
    )
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=1.0,
        help="Intervalle moyen entre paquets (s)",
    )
    parser.add_argument(
        "--mobility",
        action="store_true",
        default=False,
        help="Active la mobilité aléatoire des nœuds",
    )
    parser.add_argument(
        "--battery-capacity-j",
        type=float,
        default=DEFAULT_BATTERY_J,
        help="Capacité globale de la batterie attribuée à chaque nœud (J)",
    )
    parser.add_argument(
        "--node-capacity",
        action="append",
        default=[],
        metavar="NODE=J",
        help="Surcharge ponctuelle de capacité (ex. 3=1500). Peut être répété",
    )
    parser.add_argument(
        "--watch-nodes",
        default="",
        help="Liste d'identifiants de nœuds à surveiller (séparés par des virgules)",
    )
    parser.add_argument(
        "--stop-on-depletion",
        action="store_true",
        help="Arrête la simulation lorsque tous les nœuds surveillés sont épuisés",
    )
    args = parser.parse_args()

    node_capacity_overrides: dict[int, float] = {}
    for override in args.node_capacity:
        try:
            node_str, value_str = override.split("=", 1)
            node_capacity_overrides[int(node_str)] = float(value_str)
        except ValueError as exc:  # pragma: no cover - validation utilisateur
            raise SystemExit(
                f"Format invalide pour --node-capacity '{override}': attendu NODE=J"
            ) from exc

    watch_ids: set[int] = set()
    if args.watch_nodes:
        for raw in args.watch_nodes.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                watch_ids.add(int(raw))
            except ValueError as exc:  # pragma: no cover - validation utilisateur
                raise SystemExit(
                    f"Identifiant de nœud invalide dans --watch-nodes: '{raw}'"
                ) from exc

    records: list[dict[str, float | int | bool]] = []
    for rep in range(args.replicates):
        sim = Simulator(
            num_nodes=args.nodes,
            num_gateways=args.gateways,
            packets_to_send=args.packets,
            seed=args.seed + rep,
            battery_capacity_j=args.battery_capacity_j,
            transmission_mode=args.transmission_mode,
            packet_interval=args.packet_interval,
            mobility=args.mobility,
        )

        monitored_nodes = list(sim.nodes)
        if watch_ids:
            monitored_nodes = [n for n in sim.nodes if n.id in watch_ids]
        if args.stop_on_depletion and not monitored_nodes:
            monitored_nodes = list(sim.nodes)

        for node in sim.nodes:
            node.battery_capacity_j = args.battery_capacity_j
            if hasattr(node, "battery_remaining_j"):
                node.battery_remaining_j = args.battery_capacity_j
            override = node_capacity_overrides.get(node.id)
            if override is not None:
                node.battery_capacity_j = override
                if hasattr(node, "battery_remaining_j"):
                    node.battery_remaining_j = override

        if args.stop_on_depletion:
            monitored_ids = ", ".join(str(n.id) for n in monitored_nodes)
            if monitored_ids:
                print(f"Surveillance active sur les nœuds: {monitored_ids}")

        while sim.event_queue and sim.running:
            if not sim.step():
                break
            records.extend(_collect(sim, replicate=rep))

            if args.stop_on_depletion and monitored_nodes:
                depleted_nodes = [n for n in monitored_nodes if not n.alive]
                if depleted_nodes and all(not n.alive for n in monitored_nodes):
                    depleted_ids = ", ".join(str(n.id) for n in depleted_nodes)
                    print(
                        f"[{sim.current_time:.3f}s] Arrêt: batteries épuisées pour {depleted_ids}"
                    )
                    sim.stop()
                    break

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "battery_tracking.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
