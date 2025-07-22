#!/usr/bin/env python3
"""Calibrate simulator parameters against a FLoRa reference CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402
from VERSION_4.launcher.adr_standard_1 import apply as adr1  # noqa: E402
from VERSION_4.launcher.compare_flora import load_flora_metrics  # noqa: E402

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


def _run_sim_advanced(
    path_loss_exp: float,
    shadowing_std: float,
    fading_corr: float,
    obstacle_dB: float,
    runs: int,
    seed: int | None,
) -> dict[str, float]:
    metrics = []
    for i in range(runs):
        ch = AdvancedChannel(
            propagation_model="3d",
            fading="rayleigh",
            fading_correlation=fading_corr,
            default_obstacle_dB=obstacle_dB,
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


def _metric_error(
    sim: dict[str, float],
    flora: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Return weighted error and individual diffs for key metrics."""

    pdr_diff = abs(sim.get("PDR", 0.0) - flora.get("PDR", 0.0))
    sim_sf = sim.get("sf_distribution", {})
    flora_sf = flora.get("sf_distribution", {})
    all_sf = set(sim_sf) | set(flora_sf)
    sf_diff_abs = sum(abs(sim_sf.get(sf, 0) - flora_sf.get(sf, 0)) for sf in all_sf)
    sf_norm = max(sum(flora_sf.values()), 1)
    sf_diff = sf_diff_abs / sf_norm
    energy_diff = abs(sim.get("energy_J", 0.0) - flora.get("energy_J", 0.0))
    energy_norm = max(flora.get("energy_J", 0.0), 1.0)
    energy_rel = energy_diff / energy_norm
    err = (
        weights.get("pdr", 1.0) * pdr_diff
        + weights.get("sf", 1.0) * sf_diff
        + weights.get("energy", 1.0) * energy_rel
    )
    return err, {"pdr": pdr_diff, "sf": sf_diff_abs, "energy": energy_diff}


def calibrate(
    flora_csv: str | Path,
    runs: int = 3,
    path_loss_values: tuple[float, ...] = (2.5, 2.7, 2.9),
    shadowing_values: tuple[float, ...] = (2.0, 4.0, 6.0),
    fading_corr_values: tuple[float, ...] = (0.9,),
    obstacle_values: tuple[float, ...] = (0.0,),
    *,
    advanced: bool = False,
    seed: int | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, float] | None, float]:
    """Return best parameters minimizing metric differences."""
    if weights is None:
        weights = {"pdr": 1.0, "sf": 1.0, "energy": 1.0}
    flora = load_flora_metrics(flora_csv)
    best_err = math.inf
    best_params: dict[str, float] | None = None
    best_diffs: dict[str, float] | None = None
    for pl in path_loss_values:
        for sh in shadowing_values:
            if advanced:
                for fc in fading_corr_values:
                    for ob in obstacle_values:
                        m = _run_sim_advanced(pl, sh, fc, ob, runs, seed)
                        err, diffs = _metric_error(m, flora, weights)
                        if err < best_err:
                            best_err = err
                            best_params = {
                                "path_loss_exp": pl,
                                "shadowing_std": sh,
                                "fading_correlation": fc,
                                "obstacle_dB": ob,
                                "PDR": m.get("PDR", 0.0),
                            }
                            best_diffs = diffs
            else:
                m = _run_sim(pl, sh, runs, seed)
                err, diffs = _metric_error(m, flora, weights)
                if err < best_err:
                    best_err = err
                    best_params = {
                        "path_loss_exp": pl,
                        "shadowing_std": sh,
                        "PDR": m.get("PDR", 0.0),
                    }
                    best_diffs = diffs
    if best_params:
        diff_str = (
            f"PDR {best_diffs['pdr']:.4f}, SF diff {best_diffs['sf']}, "
            f"Energy {best_diffs['energy']:.4f} J"
        )
        print(f"Best parameters: {best_params} (error {best_err:.4f})")
        print(f"Residuals -> {diff_str}")
    return best_params, best_err


def cross_validate(
    flora_csvs: list[str | Path],
    runs: int = 3,
    path_loss_values: tuple[float, ...] = (2.5, 2.7, 2.9),
    shadowing_values: tuple[float, ...] = (2.0, 4.0, 6.0),
    fading_corr_values: tuple[float, ...] = (0.9,),
    obstacle_values: tuple[float, ...] = (0.0,),
    *,
    advanced: bool = False,
    seed: int | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, float] | None, float]:
    """Return best parameters averaged over multiple datasets."""
    if weights is None:
        weights = {"pdr": 1.0, "sf": 1.0, "energy": 1.0}
    refs = [load_flora_metrics(p) for p in flora_csvs]
    best_err = math.inf
    best_params: dict[str, float] | None = None
    best_diffs: dict[str, float] | None = None
    for pl in path_loss_values:
        for sh in shadowing_values:
            if advanced:
                for fc in fading_corr_values:
                    for ob in obstacle_values:
                        err = 0.0
                        pd = 0.0
                        sd = 0.0
                        ed = 0.0
                        for idx, flora in enumerate(refs):
                            m = _run_sim_advanced(
                                pl,
                                sh,
                                fc,
                                ob,
                                runs,
                                (seed + idx) if seed is not None else None,
                            )
                            e, diffs = _metric_error(m, flora, weights)
                            err += e
                            pd += diffs["pdr"]
                            sd += diffs["sf"]
                            ed += diffs["energy"]
                        err /= len(refs)
                        pd /= len(refs)
                        sd /= len(refs)
                        ed /= len(refs)
                        if err < best_err:
                            best_err = err
                            best_params = {
                                "path_loss_exp": pl,
                                "shadowing_std": sh,
                                "fading_correlation": fc,
                                "obstacle_dB": ob,
                            }
                            best_diffs = {"pdr": pd, "sf": sd, "energy": ed}
            else:
                err = 0.0
                pd = 0.0
                sd = 0.0
                ed = 0.0
                for idx, flora in enumerate(refs):
                    m = _run_sim(pl, sh, runs, (seed + idx) if seed is not None else None)
                    e, diffs = _metric_error(m, flora, weights)
                    err += e
                    pd += diffs["pdr"]
                    sd += diffs["sf"]
                    ed += diffs["energy"]
                err /= len(refs)
                pd /= len(refs)
                sd /= len(refs)
                ed /= len(refs)
                if err < best_err:
                    best_err = err
                    best_params = {"path_loss_exp": pl, "shadowing_std": sh}
                    best_diffs = {"pdr": pd, "sf": sd, "energy": ed}
    if best_params:
        diff_str = (
            f"PDR {best_diffs['pdr']:.4f}, SF diff {best_diffs['sf']}, "
            f"Energy {best_diffs['energy']:.4f} J"
        )
        print(f"Best parameters: {best_params} (avg error {best_err:.4f})")
        print(f"Residuals -> {diff_str}")
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
    parser.add_argument("--advanced", action="store_true", help="Use advanced channel")
    parser.add_argument("--pdr-weight", type=float, default=1.0, help="Weight for PDR difference")
    parser.add_argument("--sf-weight", type=float, default=1.0, help="Weight for SF distribution")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Weight for energy difference")
    args = parser.parse_args()
    if len(args.flora_csv) == 1:
        calibrate(
            Path(args.flora_csv[0]),
            runs=args.runs,
            seed=args.seed,
            advanced=args.advanced,
            weights={
                "pdr": args.pdr_weight,
                "sf": args.sf_weight,
                "energy": args.energy_weight,
            },
        )
    else:
        cross_validate(
            [Path(p) for p in args.flora_csv],
            runs=args.runs,
            seed=args.seed,
            advanced=args.advanced,
            weights={
                "pdr": args.pdr_weight,
                "sf": args.sf_weight,
                "energy": args.energy_weight,
            },
        )


if __name__ == "__main__":
    main()
