"""Point d'entrée pour l'étape 2."""

from __future__ import annotations

from collections import defaultdict
import csv
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

from article_c.common.csv_io import write_rows, write_simulation_results
from article_c.common.utils import (
    ensure_dir,
    parse_cli_args,
    parse_network_size_list,
    replication_ids,
    set_deterministic_seed,
    timestamp_tag,
)
from article_c.step2.simulate_step2 import run_simulation


def _aggregate_selection_probs(
    selection_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in selection_rows:
        round_id = int(row["round"])
        sf = int(row["sf"])
        selection_prob = float(row["selection_prob"])
        grouped[(round_id, sf)].append(selection_prob)
    aggregated: list[dict[str, object]] = []
    for (round_id, sf), values in sorted(grouped.items()):
        avg = sum(values) / len(values) if values else 0.0
        aggregated.append(
            {"round": round_id, "sf": sf, "selection_prob": round(avg, 6)}
        )
    return aggregated


def _aggregate_learning_curve(
    learning_curve_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in learning_curve_rows:
        network_size = int(row["network_size"])
        round_id = int(row["round"])
        algo = str(row["algo"])
        avg_reward = float(row["avg_reward"])
        grouped[(network_size, round_id, algo)].append(avg_reward)
    aggregated: list[dict[str, object]] = []
    for (network_size, round_id, algo), values in sorted(grouped.items()):
        avg = sum(values) / len(values) if values else 0.0
        aggregated.append(
            {
                "network_size": network_size,
                "round": round_id,
                "algo": algo,
                "avg_reward": round(avg, 6),
            }
        )
    return aggregated


def _log_results_written(output_dir: Path, row_count: int) -> None:
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"
    print(f"Append rows: {row_count} -> {raw_path}")
    print(f"Append rows: {row_count} -> {aggregated_path}")


def _log_unique_network_sizes(output_dir: Path) -> None:
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        print(f"Aucun raw_results.csv détecté: {raw_path}")
        return
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "network_size" not in reader.fieldnames:
            print(f"Colonne network_size absente dans {raw_path}")
            return
        values: list[float] = []
        for row in reader:
            value = row.get("network_size")
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except ValueError:
                print(f"Valeur network_size invalide détectée: {value}")
    if any(value == 0.0 for value in values):
        print(
            "ERREUR: network_size à 0.0 détecté dans raw_results.csv, vérifiez la configuration."
        )
    sizes = sorted({int(value) for value in values})
    sizes_label = ", ".join(map(str, sizes)) if sizes else "aucune"
    print(f"Tailles détectées dans raw_results: {sizes_label}")


def _simulate_density(
    task: tuple[int, int, list[int], dict[str, object], Path, Path | None]
) -> dict[str, object]:
    density, density_idx, replications, config, base_results_dir, timestamp_dir = task
    raw_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    algorithms = ("adr", "mixra_h", "mixra_opt", "ucb1_sf")
    jitter_range_s = float(config.get("jitter_range_s", 30.0))
    print(f"Jitter range utilisé (s): {jitter_range_s}")

    for replication in replications:
        seed = int(config["base_seed"]) + density_idx * 1000 + replication
        for algorithm in algorithms:
            result = run_simulation(
                algorithm=algorithm,
                density=density,
                snir_mode="snir_on",
                seed=seed,
                traffic_mode=str(config["traffic_mode"]),
                jitter_range_s=jitter_range_s,
                window_duration_s=float(config["window_duration_s"]),
                window_size=int(config["window_size"]),
                traffic_coeff_min=float(config["traffic_coeff_min"]),
                traffic_coeff_max=float(config["traffic_coeff_max"]),
                traffic_coeff_enabled=bool(config["traffic_coeff_enabled"]),
                window_delay_enabled=bool(config["window_delay_enabled"]),
                window_delay_range_s=float(config["window_delay_range_s"]),
            )
            raw_rows.extend(result.raw_rows)
            if algorithm == "ucb1_sf":
                selection_rows.extend(result.selection_prob_rows)
            learning_curve_rows.extend(result.learning_curve_rows)

    write_simulation_results(base_results_dir, raw_rows, network_size=density)
    _log_results_written(base_results_dir, len(raw_rows))
    _log_unique_network_sizes(base_results_dir)
    if timestamp_dir is not None:
        write_simulation_results(timestamp_dir, raw_rows, network_size=density)
        _log_results_written(timestamp_dir, len(raw_rows))
        _log_unique_network_sizes(timestamp_dir)
    return {
        "density": density,
        "row_count": len(raw_rows),
        "selection_rows": selection_rows,
        "learning_curve_rows": learning_curve_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    base_seed = set_deterministic_seed(args.seeds_base)
    densities = parse_network_size_list(args.network_sizes)
    replications = replication_ids(args.replications)
    simulated_sizes: list[int] = []

    base_results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(base_results_dir)
    timestamp_dir: Path | None = None
    if args.timestamp:
        timestamp_dir = base_results_dir / timestamp_tag()
        ensure_dir(timestamp_dir)

    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    total_rows = 0

    config: dict[str, object] = {
        "base_seed": base_seed,
        "traffic_mode": args.traffic_mode,
        "jitter_range_s": args.jitter_range_s,
        "window_duration_s": args.window_duration_s,
        "window_size": args.window_size,
        "traffic_coeff_min": args.traffic_coeff_min,
        "traffic_coeff_max": args.traffic_coeff_max,
        "traffic_coeff_enabled": args.traffic_coeff_enabled,
        "window_delay_enabled": args.window_delay_enabled,
        "window_delay_range_s": args.window_delay_range_s,
    }

    tasks = [
        (density, density_idx, replications, config, base_results_dir, timestamp_dir)
        for density_idx, density in enumerate(densities)
    ]

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = map(_simulate_density, tasks)
        for result in results:
            simulated_sizes.append(int(result["density"]))
            total_rows += int(result["row_count"])
            selection_rows.extend(result["selection_rows"])
            learning_curve_rows.extend(result["learning_curve_rows"])
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_simulate_density, tasks):
                simulated_sizes.append(int(result["density"]))
                total_rows += int(result["row_count"])
                selection_rows.extend(result["selection_rows"])
                learning_curve_rows.extend(result["learning_curve_rows"])

    print(f"Rows written: {total_rows}")

    if selection_rows:
        rl5_rows = _aggregate_selection_probs(selection_rows)
        rl5_path = base_results_dir / "rl5_selection_prob.csv"
        rl5_header = ["round", "sf", "selection_prob"]
        rl5_values = [[row["round"], row["sf"], row["selection_prob"]] for row in rl5_rows]
        write_rows(rl5_path, rl5_header, rl5_values)
        if timestamp_dir is not None:
            rl5_timestamp_path = timestamp_dir / "rl5_selection_prob.csv"
            write_rows(rl5_timestamp_path, rl5_header, rl5_values)

    if learning_curve_rows:
        learning_curve = _aggregate_learning_curve(learning_curve_rows)
        learning_curve_header = ["network_size", "round", "algo", "avg_reward"]
        learning_curve_values = [
            [
                row["network_size"],
                row["round"],
                row["algo"],
                row["avg_reward"],
            ]
            for row in learning_curve
        ]
        learning_curve_path = base_results_dir / "learning_curve.csv"
        write_rows(learning_curve_path, learning_curve_header, learning_curve_values)
        if timestamp_dir is not None:
            learning_curve_timestamp_path = timestamp_dir / "learning_curve.csv"
            write_rows(
                learning_curve_timestamp_path,
                learning_curve_header,
                learning_curve_values,
            )

    (base_results_dir / "done.flag").write_text("done\n", encoding="utf-8")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")


if __name__ == "__main__":
    main()
