"""Démonstration légère pour comparer SNIR activé/désactivé sur quelques fenêtres.

Le script lance deux simulations identiques (SNIR off puis on) avec un nombre
réduit de nœuds et exporte un CSV unique contenant les agrégats par cluster et
par fenêtre temporelle. Les colonnes ``window_start_s``/``window_end_s``
permettent de filtrer les lignes dans ``plot_der_by_cluster.py`` afin de
superposer les courbes dérivées et vérifier la dispersion entre états SNIR.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

from loraflexsim.launcher import Simulator  # noqa: E402

from experiments.ucb1.run_ucb1_load_sweep import (  # noqa: E402
    _assign_clusters,
    _collect_cluster_metrics,
)

DEFAULT_OUTPUT = ROOT / "experiments" / "ucb1" / "ucb1_snir_window_demo.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simule un petit scénario QoS deux fois (SNIR off/on) pour illustrer les dérivées par fenêtre."
    )
    parser.add_argument("--num-nodes", type=int, default=120, help="Nombre total de nœuds à simuler.")
    parser.add_argument(
        "--packet-interval", type=float, default=600.0, help="Intervalle d'émission en secondes."
    )
    parser.add_argument(
        "--packets-per-node", type=int, default=3, help="Nombre de paquets envoyés par nœud (définit le nombre de fenêtres)."
    )
    parser.add_argument("--seed", type=int, default=7, help="Graine aléatoire de base (décalée entre les runs SNIR off/on).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Chemin du CSV fusionné à écrire.")
    return parser.parse_args()


def _apply_snir_state(sim: Simulator, use_snir: bool) -> bool:
    """Force l'état SNIR sur le simulateur et ses canaux."""

    effective = bool(use_snir)
    setattr(sim, "use_snir", effective)
    channel = getattr(sim, "channel", None)
    if channel is not None:
        channel.use_snir = effective
    for chan in getattr(sim, "channels", []) or []:
        setattr(chan, "use_snir", effective)
    return effective


def _run_single(
    num_nodes: int,
    packet_interval: float,
    packets_per_node: int,
    seed: int,
    *,
    use_snir: bool,
) -> List[dict]:
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=1500.0,
        transmission_mode="Random",
        packet_interval=packet_interval,
        first_packet_interval=packet_interval,
        packets_to_send=packets_per_node,
        adr_node=False,
        adr_server=False,
        seed=seed,
    )
    effective_snir = _apply_snir_state(sim, use_snir)
    assignments = _assign_clusters(sim)
    sim.run()

    rows: List[dict] = []
    for entry in _collect_cluster_metrics(sim, assignments):
        payload = asdict(entry)
        payload.update(
            {
                "use_snir": effective_snir,
                "with_snir": effective_snir,
                "snir_state": "snir_on" if effective_snir else "snir_off",
            }
        )
        rows.append(payload)
    return rows


def run_demo(
    *,
    num_nodes: int = 120,
    packet_interval: float = 600.0,
    packets_per_node: int = 3,
    seed: int = 7,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    rows: list[dict] = []
    for offset, use_snir in enumerate((False, True)):
        rows.extend(
            _run_single(
                num_nodes,
                packet_interval,
                packets_per_node,
                seed + offset,
                use_snir=use_snir,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Métriques écrites dans {output_path}")


def main() -> None:
    args = parse_args()
    run_demo(
        num_nodes=args.num_nodes,
        packet_interval=args.packet_interval,
        packets_per_node=args.packets_per_node,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
