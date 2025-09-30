#!/usr/bin/env python3
"""Run the validation matrix scenarios and compare against FLoRa baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.validation import (
    SCENARIOS,
    compare_to_reference,
    load_flora_reference,
    run_validation,
)
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.gateway import Gateway, FLORA_NON_ORTH_DELTA
from loraflexsim.launcher.server import NetworkServer


def _load_example_runner():
    example_path = ROOT_DIR / "examples" / "run_flora_example.py"
    spec = importlib.util.spec_from_file_location(
        "loraflexsim.examples.run_flora_example",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le script d'exemple {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_example"):
        raise RuntimeError(
            "Le script d'exemple ne fournit pas de fonction run_example(quiet=True)"
        )
    return module.run_example


def _start_gateway_tx(
    gateway: Gateway,
    event_id: int,
    node_id: int,
    sf: int,
    rssi: float,
    start_time: float,
    end_time: float,
    *,
    frequency: float,
    capture_threshold: float,
    capture_window_symbols: int,
    capture_mode: str,
) -> dict:
    gateway.start_reception(
        event_id,
        node_id,
        sf,
        rssi,
        end_time,
        capture_threshold,
        start_time,
        frequency,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
        capture_window_symbols=capture_window_symbols,
        capture_mode=capture_mode,
    )
    _, tx = gateway.active_by_event[event_id]
    return tx


def verify_gateway_reference(reference: Path) -> bool:
    """Vérifie que la capture OMNeT++ reproduit les résultats de référence."""

    if not reference.exists():
        print(f"Gateway reference file not found: {reference}")
        return False

    with reference.open("r", encoding="utf8") as fh:
        payload = json.load(fh)

    cases = payload.get("cases", [])
    if not cases:
        print("Gateway reference is empty; skipping alignment check.")
        return False

    channel = Channel(
        phy_model="omnet",
        flora_capture=True,
        shadowing_std=0.0,
        fast_fading_std=0.0,
    )
    capture_window = channel.capture_window_symbols
    capture_threshold = channel.capture_threshold_dB

    success = True
    for idx, case in enumerate(cases, start=1):
        gw = Gateway(idx, 0.0, 0.0)
        gw.omnet_phy = channel.omnet_phy
        server = NetworkServer()
        server.gateways = [gw]

        frequency = float(case.get("frequency", 868e6))
        preambles = case.get("preamble")
        count = len(case.get("rssi", []))

        for event_id, (rssi, start, end, sf) in enumerate(
            zip(
                case.get("rssi", []),
                case.get("start", []),
                case.get("end", []),
                case.get("sf", []),
            ),
            start=1,
        ):
            tx = _start_gateway_tx(
                gw,
                event_id,
                event_id,
                int(sf),
                float(rssi),
                float(start),
                float(end),
                frequency=frequency,
                capture_threshold=capture_threshold,
                capture_window_symbols=capture_window,
                capture_mode="omnet",
            )
            if preambles:
                tx["preamble_symbols"] = preambles[event_id - 1]

        winners = [
            not gw.active_by_event[event_id][1]["lost_flag"]
            for event_id in range(1, count + 1)
        ]
        expected = [bool(value) for value in case.get("expected", [])]
        if winners != expected:
            print(
                "gateway_case_%d mismatch: expected %s got %s"
                % (idx, expected, winners)
            )
            success = False
        else:
            print(f"gateway_case_{idx}: ok")

        for event_id in range(1, count + 1):
            gw.end_reception(event_id, server, event_id)

    return success


def run_matrix(output: Path, repeat: int) -> bool:
    """Execute all validation scenarios and persist a summary CSV."""

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    overall_success = True

    gateway_reference = ROOT_DIR / "tests" / "data" / "flora_gateway_reference.json"
    if not verify_gateway_reference(gateway_reference):
        overall_success = False

    try:
        run_example = _load_example_runner()
        example_metrics = run_example(quiet=True)
    except Exception as exc:  # pragma: no cover - dépend des extras Panel
        print(f"Example scenario failed: {exc}")
        overall_success = False
    else:
        rows.append(
            {
                "scenario": "flora_example",
                "description": "Exemple FLoRa n100-gw1.ini",
                "pdr_sim": float(example_metrics.get("PDR", 0.0)),
                "pdr_ref": float(example_metrics.get("PDR", 0.0)),
                "pdr_delta": 0.0,
                "pdr_std": 0.0,
                "collisions_sim": float(example_metrics.get("collisions", 0.0)),
                "collisions_ref": float(example_metrics.get("collisions", 0.0)),
                "collisions_delta": 0.0,
                "snr_sim": float(example_metrics.get("snr", 0.0)),
                "snr_ref": float(example_metrics.get("snr", 0.0)),
                "snr_delta": 0.0,
                "tolerance_pdr": 0.0,
                "tolerance_collisions": 0.0,
                "tolerance_snr": 0.0,
                "status": "example",
            }
        )
        print("flora_example: métriques collectées et ajoutées au CSV")

    for scenario in SCENARIOS:
        run_count = repeat if scenario.name == "long_range" else 1
        metrics_runs: list[dict[str, float]] = []
        for idx in range(run_count):
            overrides: dict[str, float | int | None] = {}
            if scenario.name == "long_range" and run_count > 1:
                base_seed = scenario.sim_kwargs.get("seed")
                if base_seed is not None:
                    overrides["seed"] = int(base_seed) + idx
            sim = scenario.build_simulator(**overrides)
            metrics_runs.append(run_validation(sim, scenario.run_steps))

        metrics = {
            key: sum(run[key] for run in metrics_runs) / len(metrics_runs)
            for key in metrics_runs[0]
        }
        pdr_std = (
            statistics.pstdev(run["PDR"] for run in metrics_runs)
            if len(metrics_runs) > 1
            else 0.0
        )
        reference = load_flora_reference(scenario.flora_reference)
        deltas = compare_to_reference(metrics, reference, scenario.tolerances)

        status = (
            deltas["PDR"] <= scenario.tolerances.pdr
            and deltas["collisions"] <= scenario.tolerances.collisions
            and deltas["snr"] <= scenario.tolerances.snr
        )
        overall_success &= status

        rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "pdr_sim": metrics["PDR"],
                "pdr_ref": reference["PDR"],
                "pdr_delta": deltas["PDR"],
                "pdr_std": pdr_std,
                "collisions_sim": metrics["collisions"],
                "collisions_ref": reference["collisions"],
                "collisions_delta": deltas["collisions"],
                "snr_sim": metrics["snr"],
                "snr_ref": reference["snr"],
                "snr_delta": deltas["snr"],
                "tolerance_pdr": scenario.tolerances.pdr,
                "tolerance_collisions": scenario.tolerances.collisions,
                "tolerance_snr": scenario.tolerances.snr,
                "status": "ok" if status else "fail",
            }
        )

    fieldnames: Sequence[str] = rows[0].keys() if rows else []
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        pdr_value = f"{row['pdr_sim']:.3f}"
        if row.get("pdr_std", 0.0):
            pdr_value += f"±{row['pdr_std']:.3f}"
        print(
            f"{row['scenario']}: PDR={pdr_value} (ref {row['pdr_ref']:.3f}, Δ{row['pdr_delta']:.3f}) | "
            f"Collisions={row['collisions_sim']:.1f} (ref {row['collisions_ref']:.1f}, Δ{row['collisions_delta']:.1f}) | "
            f"SNR={row['snr_sim']:.2f} dB (ref {row['snr_ref']:.2f}, Δ{row['snr_delta']:.2f}) -> {row['status']}"
        )

    print(f"Summary written to {output}")
    return overall_success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/validation_matrix.csv"),
        help="Destination CSV file for the summary table.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Nombre d'exécutions pour le scénario long_range afin de calculer la "
            "moyenne et l'écart-type du PDR."
        ),
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat doit être supérieur ou égal à 1")

    try:
        success = run_matrix(args.output, args.repeat)
    except RuntimeError as exc:  # pragma: no cover - propagates optional deps errors
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
