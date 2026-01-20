"""Point d'entrée pour l'étape 2."""

from __future__ import annotations

from collections import defaultdict
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


def _log_results_written(output_dir: Path, row_count: int) -> None:
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"
    print(f"Append rows: {row_count} -> {raw_path}")
    print(f"Append rows: {row_count} -> {aggregated_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    base_seed = set_deterministic_seed(args.seeds_base)
    densities = parse_network_size_list(args.network_sizes)
    replications = replication_ids(args.replications)
    snir_mode = "snir_on"

    base_results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(base_results_dir)
    timestamp_dir: Path | None = None
    if args.timestamp:
        timestamp_dir = base_results_dir / timestamp_tag()
        ensure_dir(timestamp_dir)

    raw_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    algorithms = ("adr", "mixra_h", "mixra_opt", "ucb1_sf")

    for density_idx, density in enumerate(densities):
        for replication in replications:
            seed = base_seed + density_idx * 1000 + replication
            for algorithm in algorithms:
                result = run_simulation(
                    algorithm=algorithm,
                    density=density,
                    snir_mode=snir_mode,
                    seed=seed,
                    traffic_mode=args.traffic_mode,
                    jitter_range_s=args.jitter_range,
                    window_duration_s=args.window_duration_s,
                    window_size=args.window_size,
                    traffic_coeff_min=args.traffic_coeff_min,
                    traffic_coeff_max=args.traffic_coeff_max,
                    traffic_coeff_enabled=args.traffic_coeff_enabled,
                    window_delay_enabled=args.window_delay_enabled,
                    window_delay_range_s=args.window_delay_range_s,
                )
                raw_rows.extend(result.raw_rows)
                if algorithm == "ucb1_sf":
                    selection_rows.extend(result.selection_prob_rows)

    write_simulation_results(base_results_dir, raw_rows)
    _log_results_written(base_results_dir, len(raw_rows))
    if timestamp_dir is not None:
        write_simulation_results(timestamp_dir, raw_rows)
        _log_results_written(timestamp_dir, len(raw_rows))
    print(f"Rows written: {len(raw_rows)}")

    if selection_rows:
        rl5_rows = _aggregate_selection_probs(selection_rows)
        rl5_path = base_results_dir / "rl5_selection_prob.csv"
        rl5_header = ["round", "sf", "selection_prob"]
        rl5_values = [[row["round"], row["sf"], row["selection_prob"]] for row in rl5_rows]
        write_rows(rl5_path, rl5_header, rl5_values)
        if timestamp_dir is not None:
            rl5_timestamp_path = timestamp_dir / "rl5_selection_prob.csv"
            write_rows(rl5_timestamp_path, rl5_header, rl5_values)

    (base_results_dir / "done.flag").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
