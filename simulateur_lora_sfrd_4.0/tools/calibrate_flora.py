#!/usr/bin/env python3
"""Calibrate simulator parameters against a FLoRa reference CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.simulator import Simulator
from VERSION_4.launcher.channel import Channel
from VERSION_4.launcher.adr_standard_1 import apply as adr1
from VERSION_4.launcher.compare_flora import load_flora_metrics

NODE_POSITIONS = [
    (450.45, 490.0),
    (555.0, 563.0),
    (460.0, 467.0),
    (565.0, 571.0),
    (702.0, 578.0),
]

GW_POSITION = (500.0, 500.0)


def _run_sim(path_loss_exp: float, shadowing_std: float, runs: int, seed: int | None) -> dict[str, float]:
    metrics = []
    for i in range(runs):
        ch = Channel(
            path_loss_exp=path_loss_exp,
            shadowing_std=shadowing_std,
            detection_threshold_dBm=-110.0,
        )
        sim = Simulator(
            num_nodes=5,
            num_gateways=1,
            area_size=1000.0,
            transmission_mode="Random",
            packet_interval=100.0,
            packets_to_send=80,
            adr_node=True,
            adr_server=True,
            mobility=False,
            fixed_sf=12,
            fixed_tx_power=14.0,
            channels=[ch],
            detection_threshold_dBm=-110.0,
            min_interference_time=5.0,
            seed=(seed + i) if seed is not None else None,
        )
        adr1(sim)
        gw = sim.gateways[0]
        gw.x, gw.y = GW_POSITION
        for node, pos in zip(sim.nodes, NODE_POSITIONS):
            node.x, node.y = pos
            node.initial_x, node.initial_y = pos
        sim.run()
        metrics.append(sim.get_metrics())
    avg: dict[str, float] = {}
    for k in metrics[0]:
        if isinstance(metrics[0][k], (int, float)):
            avg[k] = sum(m[k] for m in metrics) / len(metrics)
    return avg


def calibrate(
    flora_csv: str | Path,
    runs: int = 3,
    path_loss_values: tuple[float, ...] = (2.5, 2.7, 2.9),
    shadowing_values: tuple[float, ...] = (2.0, 4.0, 6.0),
    *,
    seed: int | None = None,
) -> tuple[dict[str, float] | None, float]:
    """Return best parameters minimizing the PDR difference."""
    flora = load_flora_metrics(flora_csv)
    best_err = math.inf
    best_params: dict[str, float] | None = None
    for pl in path_loss_values:
        for sh in shadowing_values:
            m = _run_sim(pl, sh, runs, seed)
            err = abs(m.get("PDR", 0.0) - flora.get("PDR", 0.0))
            if err < best_err:
                best_err = err
                best_params = {"path_loss_exp": pl, "shadowing_std": sh, "PDR": m.get("PDR", 0.0)}
    if best_params:
        print(f"Best parameters: {best_params} (PDR diff {best_err:.4f})")
    return best_params, best_err


def cross_validate(
    flora_csvs: list[str | Path],
    runs: int = 3,
    path_loss_values: tuple[float, ...] = (2.5, 2.7, 2.9),
    shadowing_values: tuple[float, ...] = (2.0, 4.0, 6.0),
    *,
    seed: int | None = None,
) -> tuple[dict[str, float] | None, float]:
    """Return best parameters averaged over multiple datasets."""
    refs = [load_flora_metrics(p) for p in flora_csvs]
    best_err = math.inf
    best_params: dict[str, float] | None = None
    for pl in path_loss_values:
        for sh in shadowing_values:
            err = 0.0
            for idx, flora in enumerate(refs):
                m = _run_sim(pl, sh, runs, (seed + idx) if seed is not None else None)
                pdr_diff = abs(m.get("PDR", 0.0) - flora.get("PDR", 0.0))
                sim_sf = m.get("sf_distribution", {})
                flora_sf = flora.get("sf_distribution", {})
                all_sf = set(sim_sf) | set(flora_sf)
                sf_diff = sum(abs(sim_sf.get(sf, 0) - flora_sf.get(sf, 0)) for sf in all_sf)
                norm = max(sum(flora_sf.values()), 1)
                err += pdr_diff + sf_diff / norm
            err /= len(refs)
            if err < best_err:
                best_err = err
                best_params = {"path_loss_exp": pl, "shadowing_std": sh}
    if best_params:
        print(f"Best parameters: {best_params} (avg error {best_err:.4f})")
    return best_params, best_err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate simulator using one or more FLoRa CSV references"
    )
    parser.add_argument(
        "flora_csv",
        nargs="+",
        help="CSV exported from FLoRa (can provide several for cross-validation)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per parameter set")
    parser.add_argument("--seed", type=int, help="Base random seed")
    args = parser.parse_args()
    if len(args.flora_csv) == 1:
        calibrate(Path(args.flora_csv[0]), runs=args.runs, seed=args.seed)
    else:
        cross_validate([Path(p) for p in args.flora_csv], runs=args.runs, seed=args.seed)


if __name__ == "__main__":
    main()
