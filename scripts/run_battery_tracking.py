"""Run a small simulation and track battery levels after each event.

This script executes the simulator step by step, collecting the remaining
energy of each node after every processed event.  The collected data is stored
in ``results/battery_tracking.csv`` with columns ``time``, ``node_id``,
``energy_j``, ``capacity_j``, ``alive`` and ``replicate``.  Multiple replicates
can be executed to gather statistics across runs.

Usage::

    python scripts/run_battery_tracking.py --nodes 5 --packets 3 --seed 1 --replicates 2
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

# Limite du nombre d'enregistrements écrits par nœud et par réplicat.  Cette
# réduction permet d'éviter des fichiers inutilement volumineux lorsque la
# simulation progresse événement par événement, tout en conservant une
# représentation fidèle de la tendance énergétique (premier/dernier échantillon
# et points régulièrement espacés).
MAX_RECORDS_PER_NODE = 40


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
            packets_to_send=args.packets,
            seed=args.seed + rep,
            battery_capacity_j=args.battery_capacity_j,
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

    df = pd.DataFrame(records)
    if not df.empty:
        # Conserver au maximum ``MAX_RECORDS_PER_NODE`` lignes par paire
        # (réplicat, nœud) en préservant les points clés : début, fin,
        # transitions d'état et instants d'épuisement.
        def _downsample(group: pd.DataFrame) -> pd.DataFrame:
            length = len(group)
            if length <= MAX_RECORDS_PER_NODE:
                return group
            if MAX_RECORDS_PER_NODE <= 2:
                return group.iloc[[0, -1]]

            # Indices à conserver en priorité
            important: set[int] = {0, length - 1}

            if "alive" in group.columns:
                alive_values = group["alive"].astype(bool).tolist()
                for idx in range(1, length):
                    if alive_values[idx] != alive_values[idx - 1]:
                        important.update({idx - 1, idx})

            if "energy_j" in group.columns:
                for idx, energy in enumerate(group["energy_j"].tolist()):
                    if energy <= 0:
                        important.add(idx)

            # Si trop de points critiques, tronquer en conservant l'ordre.
            if len(important) >= MAX_RECORDS_PER_NODE:
                selected = sorted(important)[:MAX_RECORDS_PER_NODE]
                return group.iloc[selected]

            # Compléter avec un échantillonnage régulier pour représenter la tendance.
            remaining_slots = MAX_RECORDS_PER_NODE - len(important)
            if remaining_slots > 0:
                step = (length - 1) / (remaining_slots + 1)
                for i in range(1, remaining_slots + 1):
                    important.add(round(i * step))
                    if len(important) >= MAX_RECORDS_PER_NODE:
                        break

            # En cas de doublons dûs à l'arrondi, compléter séquentiellement.
            idx = 0
            limit = min(MAX_RECORDS_PER_NODE, length)
            while len(important) < limit and idx < length:
                if idx not in important:
                    important.add(idx)
                idx += 1

            selected = sorted(important)[:MAX_RECORDS_PER_NODE]
            return group.iloc[selected]

        grouped: list[pd.DataFrame] = []
        for _, group in df.groupby(["replicate", "node_id"], sort=False):
            grouped.append(_downsample(group.reset_index(drop=True)))
        df = (
            pd.concat(grouped, ignore_index=True)
            .sort_values(["replicate", "node_id", "time"])
            .reset_index(drop=True)
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "battery_tracking.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
