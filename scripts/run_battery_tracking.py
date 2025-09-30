"""Run a small simulation and track battery levels after each event.

This script executes the simulator step by step, collecting the remaining
energy of each node after every processed event.  The collected data is stored
in ``results/battery_tracking.csv`` with columns ``time``, ``node_id``,
``energy_j``, ``capacity_j`` and ``replicate``.  Multiple replicates can be
executed to gather statistics across runs.

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


# Default battery capacity in joules for each node.  A finite value is required
# to observe the remaining energy decreasing over time.
DEFAULT_BATTERY_J = 1000.0


def _collect(sim: Simulator, replicate: int) -> Iterable[dict[str, float | int]]:
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
    args = parser.parse_args()

    records: list[dict[str, float | int]] = []
    for rep in range(args.replicates):
        sim = Simulator(
            num_nodes=args.nodes,
            packets_to_send=args.packets,
            seed=args.seed + rep,
            battery_capacity_j=DEFAULT_BATTERY_J,
        )

        while sim.event_queue and sim.running:
            sim.run(max_steps=1)  # Process one event at a time
            records.extend(_collect(sim, replicate=rep))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "battery_tracking.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
