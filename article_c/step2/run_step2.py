"""Point d'entrée pour l'étape 2."""

from __future__ import annotations

from collections import defaultdict
import csv
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from typing import Sequence

from article_c.common.csv_io import write_rows, write_simulation_results
from article_c.common.plot_helpers import apply_plot_style, place_legend, save_figure
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
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in selection_rows:
        network_size = int(row["network_size"])
        round_id = int(row["round"])
        sf = int(row["sf"])
        selection_prob = float(row["selection_prob"])
        grouped[(network_size, round_id, sf)].append(selection_prob)
    aggregated: list[dict[str, object]] = []
    for (network_size, round_id, sf), values in sorted(grouped.items()):
        avg = sum(values) / len(values) if values else 0.0
        aggregated.append(
            {
                "network_size": network_size,
                "density": network_size,
                "round": round_id,
                "sf": sf,
                "selection_prob": round(avg, 6),
            }
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
                "density": network_size,
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


def _summarize_success_collision(
    raw_rows: list[dict[str, object]]
) -> dict[str, float]:
    success_rates: list[float] = []
    collision_norms: list[float] = []
    for row in raw_rows:
        if "success_rate" in row:
            success_rates.append(float(row["success_rate"]))
        if "collision_norm" in row:
            collision_norms.append(float(row["collision_norm"]))
    if not success_rates:
        success_rates = [0.0]
    if not collision_norms:
        collision_norms = [0.0]
    return {
        "success_min": min(success_rates),
        "success_max": max(success_rates),
        "success_mean": sum(success_rates) / len(success_rates),
        "collision_min": min(collision_norms),
        "collision_max": max(collision_norms),
        "collision_mean": sum(collision_norms) / len(collision_norms),
    }


def _log_size_diagnostics(density: int, metrics: dict[str, float]) -> None:
    print(
        "Diagnostic taille "
        f"{density}: succès min/max = {metrics['success_min']:.4f}/"
        f"{metrics['success_max']:.4f}, "
        f"collisions min/max = {metrics['collision_min']:.4f}/"
        f"{metrics['collision_max']:.4f}"
    )


def _verify_metric_variation(size_metrics: dict[int, dict[str, float]]) -> None:
    if len(size_metrics) < 2:
        return
    success_means = {round(metrics["success_mean"], 6) for metrics in size_metrics.values()}
    collision_means = {
        round(metrics["collision_mean"], 6) for metrics in size_metrics.values()
    }
    if len(success_means) == 1:
        print(
            "ERREUR: le success_rate moyen ne varie pas avec la taille du réseau."
        )
    if len(collision_means) == 1:
        print(
            "ERREUR: les collisions moyennes ne varient pas avec la taille du réseau."
        )


def _read_aggregated_sizes(aggregated_path: Path) -> set[int]:
    if not aggregated_path.exists():
        print(f"Aucun aggregated_results.csv détecté: {aggregated_path}")
        return set()
    with aggregated_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        preview = ", ".join(fieldnames[:6]) if fieldnames else "aucun"
        print(f"Premiers headers lus dans {aggregated_path}: {preview}")
        size_key = None
        if "network_size" in fieldnames:
            size_key = "network_size"
        elif "density" in fieldnames:
            size_key = "density"
            print(
                f"Colonne network_size absente dans {aggregated_path}, "
                "fallback sur density."
            )
        else:
            print(
                f"Colonnes network_size/density absentes dans {aggregated_path}"
            )
            return set()
        sizes: set[int] = set()
        for row in reader:
            value = row.get(size_key)
            if value in (None, ""):
                continue
            try:
                sizes.add(int(float(value)))
            except ValueError:
                print(
                    f"Valeur {size_key} invalide détectée: {value}"
                )
        return sizes


def _is_non_empty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_step2_aggregated_with_errors(
    aggregated_path: Path,
) -> list[dict[str, object]]:
    if not aggregated_path.exists():
        print(f"Aucun aggregated_results.csv détecté: {aggregated_path}")
        return []
    rows: list[dict[str, object]] = []
    with aggregated_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            network_size_value = row.get("network_size") or row.get("density")
            if network_size_value in (None, ""):
                continue
            rows.append(
                {
                    "network_size": int(float(network_size_value)),
                    "algo": str(row.get("algo", "")),
                    "snir_mode": str(row.get("snir_mode", "")),
                    "cluster": str(row.get("cluster", "all")),
                    "reward_mean": float(row.get("reward_mean", 0.0) or 0.0),
                    "reward_std": float(row.get("reward_std", 0.0) or 0.0),
                    "reward_ci95": float(row.get("reward_ci95", 0.0) or 0.0),
                }
            )
    return rows


def _plot_summary_reward(output_dir: Path) -> None:
    aggregated_path = output_dir / "aggregated_results.csv"
    rows = _load_step2_aggregated_with_errors(aggregated_path)
    if not rows:
        print("Aucune ligne agrégée disponible pour le plot de synthèse.")
        return
    rows = [
        row
        for row in rows
        if row.get("cluster") == "all" and row.get("snir_mode") == "snir_on"
    ]
    if not rows:
        print("Aucune ligne agrégée filtrée pour le plot de synthèse.")
        return
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    network_sizes = sorted({row["network_size"] for row in rows})
    algorithms = sorted({row["algo"] for row in rows})
    error_key = "reward_ci95" if any(row.get("reward_ci95") for row in rows) else "reward_std"
    for algo in algorithms:
        algo_rows = [row for row in rows if row["algo"] == algo]
        points = {row["network_size"]: row["reward_mean"] for row in algo_rows}
        errors = {row["network_size"]: row.get(error_key, 0.0) for row in algo_rows}
        values = [points.get(size, float("nan")) for size in network_sizes]
        yerr = [errors.get(size, 0.0) for size in network_sizes]
        ax.errorbar(
            network_sizes,
            values,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=algo,
        )
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Step 2 - Reward moyen (avec barres d'erreur)")
    ax.set_xticks(network_sizes)
    place_legend(ax)
    output_plot_dir = output_dir / "plots"
    save_figure(fig, output_plot_dir, "summary_reward", use_tight=False)
    plt.close(fig)


def _simulate_density(
    task: tuple[int, int, list[int], dict[str, object], Path, Path | None]
) -> dict[str, object]:
    density, density_idx, replications, config, base_results_dir, timestamp_dir = task
    raw_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    algorithms = ("adr", "mixra_h", "mixra_opt", "ucb1_sf")
    algo_offsets = {algo: idx * 10000 for idx, algo in enumerate(algorithms)}
    jitter_range_s = float(config.get("jitter_range_s", 30.0))
    print(f"Jitter range utilisé (s): {jitter_range_s}")
    offsets_label = ", ".join(
        f"{algo}={offset}" for algo, offset in algo_offsets.items()
    )
    print(f"Offsets de seed par algorithme: {offsets_label}")

    for replication in replications:
        seed = int(config["base_seed"]) + density_idx * 1000 + replication
        for algorithm in algorithms:
            algorithm_seed = seed + algo_offsets[algorithm]
            result = run_simulation(
                algorithm=algorithm,
                n_nodes=int(density),
                density=density,
                snir_mode="snir_on",
                seed=algorithm_seed,
                traffic_mode=str(config["traffic_mode"]),
                jitter_range_s=jitter_range_s,
                window_duration_s=float(config["window_duration_s"]),
                window_size=int(config["window_size"]),
                traffic_coeff_min=float(config["traffic_coeff_min"]),
                traffic_coeff_max=float(config["traffic_coeff_max"]),
                traffic_coeff_enabled=bool(config["traffic_coeff_enabled"]),
                window_delay_enabled=bool(config["window_delay_enabled"]),
                window_delay_range_s=float(config["window_delay_range_s"]),
                reference_network_size=int(config["reference_network_size"]),
            )
            for row in result.raw_rows:
                row["replication"] = replication
            raw_rows.extend(result.raw_rows)
            if algorithm == "ucb1_sf":
                selection_rows.extend(result.selection_prob_rows)
            learning_curve_rows.extend(result.learning_curve_rows)

    invalid_sizes = {
        int(row.get("network_size", -1))
        for row in raw_rows
        if int(row.get("network_size", -1)) != int(density)
    }
    if invalid_sizes:
        raise ValueError(
            "network_size différent de n_nodes détecté pour la taille "
            f"{density}: {sorted(invalid_sizes)}"
        )
    diagnostics = _summarize_success_collision(raw_rows)
    _log_size_diagnostics(int(density), diagnostics)

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
        "diagnostics": diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    base_seed = set_deterministic_seed(args.seeds_base)
    densities = parse_network_size_list(args.network_sizes)
    requested_sizes = list(densities)
    reference_network_size = (
        int(args.reference_network_size)
        if getattr(args, "reference_network_size", None) is not None
        else int(round(median(requested_sizes)))
    )
    replications = replication_ids(args.replications)
    simulated_sizes: list[int] = []

    base_results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(base_results_dir)
    timestamp_dir: Path | None = None
    if args.timestamp:
        timestamp_dir = base_results_dir / timestamp_tag()
        ensure_dir(timestamp_dir)
    aggregated_path = base_results_dir / "aggregated_results.csv"
    if _is_non_empty_file(aggregated_path):
        size_bytes = aggregated_path.stat().st_size
        print(
            "aggregated_results.csv existant détecté "
            f"({size_bytes} octets) : aucune réinitialisation."
        )

    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    size_diagnostics: dict[int, dict[str, float]] = {}
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
        "reference_network_size": max(1, reference_network_size),
    }

    aggregated_sizes = _read_aggregated_sizes(base_results_dir / "aggregated_results.csv")
    requested_set = set(requested_sizes)
    existing_sizes = sorted(requested_set & aggregated_sizes)
    remaining_sizes = sorted(requested_set - aggregated_sizes)
    existing_label = ", ".join(map(str, existing_sizes)) if existing_sizes else "aucune"
    if args.resume:
        densities = remaining_sizes
        print("Mode reprise activé: exclusion des tailles déjà agrégées.")
    simulated_targets = densities if args.resume else requested_sizes
    simulated_label = (
        ", ".join(map(str, simulated_targets)) if simulated_targets else "aucune"
    )
    print(f"Tailles déjà présentes dans aggregated_results.csv: {existing_label}")
    print(f"Tailles à simuler: {simulated_label}")

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
            size_diagnostics[int(result["density"])] = dict(result["diagnostics"])
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_simulate_density, tasks):
                simulated_sizes.append(int(result["density"]))
                total_rows += int(result["row_count"])
                selection_rows.extend(result["selection_rows"])
                learning_curve_rows.extend(result["learning_curve_rows"])
                size_diagnostics[int(result["density"])] = dict(result["diagnostics"])

    print(f"Rows written: {total_rows}")
    _verify_metric_variation(size_diagnostics)

    if selection_rows:
        rl5_rows = _aggregate_selection_probs(selection_rows)
        rl5_path = base_results_dir / "rl5_selection_prob.csv"
        rl5_header = ["network_size", "density", "round", "sf", "selection_prob"]
        rl5_values = [
            [
                row["network_size"],
                row["density"],
                row["round"],
                row["sf"],
                row["selection_prob"],
            ]
            for row in rl5_rows
        ]
        write_rows(rl5_path, rl5_header, rl5_values)
        if timestamp_dir is not None:
            rl5_timestamp_path = timestamp_dir / "rl5_selection_prob.csv"
            write_rows(rl5_timestamp_path, rl5_header, rl5_values)

    if learning_curve_rows:
        learning_curve = _aggregate_learning_curve(learning_curve_rows)
        learning_curve_header = ["network_size", "density", "round", "algo", "avg_reward"]
        learning_curve_values = [
            [
                row["network_size"],
                row["density"],
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

    aggregated_sizes = _read_aggregated_sizes(base_results_dir / "aggregated_results.csv")
    missing_sizes = sorted(set(requested_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        print(
            "ATTENTION: tailles manquantes dans aggregated_results.csv, "
            f"done.flag non écrit. Manquantes: {missing_label}"
        )
    else:
        (base_results_dir / "done.flag").write_text("done\n", encoding="utf-8")
        print("done.flag écrit (agrégation complète).")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")
    if args.plot_summary:
        _plot_summary_reward(base_results_dir)


if __name__ == "__main__":
    main()
