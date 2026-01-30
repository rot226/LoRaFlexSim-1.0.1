"""Point d'entrée pour l'étape 2."""

from __future__ import annotations

from collections import defaultdict
import csv
import logging
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from typing import Sequence

from article_c.common.csv_io import write_rows, write_simulation_results
from article_c.common.plot_helpers import (
    apply_plot_style,
    parse_export_formats,
    place_legend,
    save_figure,
    set_default_export_formats,
)
from article_c.common.utils import (
    ensure_dir,
    parse_cli_args,
    parse_network_size_list,
    replication_ids,
    set_deterministic_seed,
    timestamp_tag,
)
from article_c.step2.simulate_step2 import run_simulation
from plot_defaults import DEFAULT_FIGSIZE_MULTI


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


def _assert_flat_output_files(output_dir: Path, density: int | float) -> None:
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"
    missing = [str(path) for path in (raw_path, aggregated_path) if not path.exists()]
    if missing:
        missing_label = ", ".join(missing)
        message = (
            "ERREUR: fichiers de sortie attendus absents après "
            f"write_simulation_results pour la taille {density}: {missing_label}"
        )
        print(message)
        raise FileNotFoundError(message)


def _log_unique_network_sizes(output_dir: Path) -> None:
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        print(f"Aucun raw_results.csv détecté: {raw_path}")
        return
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers_label = ", ".join(reader.fieldnames or [])
        print(f"Headers lus dans {raw_path}: {headers_label or 'aucun'}")
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
    reward_values: list[float] = []
    throughput_success_values: list[float] = []
    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        if "success_rate" in row:
            success_rates.append(float(row["success_rate"]))
        if "collision_norm" in row:
            collision_norms.append(float(row["collision_norm"]))
        if "reward" in row:
            reward_values.append(float(row["reward"]))
        if "throughput_success" in row:
            throughput_success_values.append(float(row["throughput_success"]))
    if not success_rates:
        success_rates = [0.0]
    if not collision_norms:
        collision_norms = [0.0]
    if not reward_values:
        reward_values = [0.0]
    if not throughput_success_values:
        throughput_success_values = [0.0]
    return {
        "success_min": min(success_rates),
        "success_max": max(success_rates),
        "success_mean": sum(success_rates) / len(success_rates),
        "collision_min": min(collision_norms),
        "collision_max": max(collision_norms),
        "collision_mean": sum(collision_norms) / len(collision_norms),
        "reward_mean": sum(reward_values) / len(reward_values),
        "throughput_success_mean": sum(throughput_success_values)
        / len(throughput_success_values),
    }


def _assert_flat_output_sizes(
    base_results_dir: Path, simulated_sizes: list[int]
) -> None:
    aggregated_sizes = _read_aggregated_sizes(
        base_results_dir / "aggregated_results.csv"
    )
    missing_sizes = sorted(set(simulated_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        message = (
            "ERREUR: write_simulation_results manquant pour certaines tailles "
            f"simulées (flat_output=True): {missing_label}"
        )
        print(message)
        raise RuntimeError(message)


def _init_collision_histogram() -> dict[str, int]:
    return {"0-0.1": 0, "0.1-0.3": 0, "0.3-0.6": 0, "0.6-1.0": 0}


def _update_collision_histogram(histogram: dict[str, int], value: float) -> None:
    bounded = max(0.0, min(1.0, value))
    if bounded < 0.1:
        histogram["0-0.1"] += 1
    elif bounded < 0.3:
        histogram["0.1-0.3"] += 1
    elif bounded < 0.6:
        histogram["0.3-0.6"] += 1
    else:
        histogram["0.6-1.0"] += 1


def _summarize_post_simulation(
    raw_rows: list[dict[str, object]]
) -> dict[str, object]:
    success_sum = 0.0
    success_count = 0
    success_zero_count = 0
    collision_sum = 0.0
    collision_count = 0
    collision_hist = _init_collision_histogram()
    link_quality_sum = 0.0
    link_quality_count = 0
    link_quality_min: float | None = None
    link_quality_max: float | None = None
    reward_zero_no_success = 0
    reward_zero_clipped = 0
    reward_zero_total = 0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_count = 0
    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        success_rate = float(row.get("success_rate", 0.0) or 0.0)
        collision_norm = float(row.get("collision_norm", 0.0) or 0.0)
        link_quality = float(row.get("link_quality", 0.0) or 0.0)
        reward = float(row.get("reward", 0.0) or 0.0)
        success_sum += success_rate
        success_count += 1
        if success_rate <= 1e-9:
            success_zero_count += 1
        collision_sum += collision_norm
        collision_count += 1
        _update_collision_histogram(collision_hist, collision_norm)
        link_quality_sum += link_quality
        link_quality_count += 1
        link_quality_min = (
            link_quality if link_quality_min is None else min(link_quality_min, link_quality)
        )
        link_quality_max = (
            link_quality if link_quality_max is None else max(link_quality_max, link_quality)
        )
        reward_min = reward if reward_min is None else min(reward_min, reward)
        reward_max = reward if reward_max is None else max(reward_max, reward)
        reward_count += 1
        if reward <= 1e-9:
            reward_zero_total += 1
            if success_rate <= 1e-9:
                reward_zero_no_success += 1
            else:
                reward_zero_clipped += 1
    return {
        "success_sum": success_sum,
        "success_count": success_count,
        "success_zero_count": success_zero_count,
        "collision_sum": collision_sum,
        "collision_count": collision_count,
        "collision_hist": collision_hist,
        "link_quality_sum": link_quality_sum,
        "link_quality_count": link_quality_count,
        "link_quality_min": 0.0 if link_quality_min is None else link_quality_min,
        "link_quality_max": 0.0 if link_quality_max is None else link_quality_max,
        "reward_zero_no_success": reward_zero_no_success,
        "reward_zero_clipped": reward_zero_clipped,
        "reward_zero_total": reward_zero_total,
        "reward_min": 0.0 if reward_min is None else reward_min,
        "reward_max": 0.0 if reward_max is None else reward_max,
        "reward_count": reward_count,
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

    def _has_variation(
        values: list[float], rel_tol: float = 1e-6, abs_tol: float = 1e-9
    ) -> bool:
        if len(values) < 2:
            return False
        min_value = min(values)
        max_value = max(values)
        span = max_value - min_value
        scale = max(abs(max_value), abs(min_value), abs_tol)
        return span > max(abs_tol, scale * rel_tol)

    success_means = [
        metrics.get("success_mean", 0.0) for metrics in size_metrics.values()
    ]
    collision_means = [
        metrics.get("collision_mean", 0.0) for metrics in size_metrics.values()
    ]
    reward_means = [
        metrics.get("reward_mean", 0.0) for metrics in size_metrics.values()
    ]
    throughput_means = [
        metrics.get("throughput_success_mean", 0.0) for metrics in size_metrics.values()
    ]
    if not _has_variation(success_means):
        print(
            "ERREUR: le success_rate moyen ne varie pas avec la taille du réseau."
        )
    if not _has_variation(collision_means):
        print(
            "ERREUR: les collisions moyennes ne varient pas avec la taille du réseau."
        )
    if not _has_variation(reward_means):
        print(
            "ERREUR: le reward_mean moyen ne varie pas avec la taille du réseau."
        )
    if not _has_variation(throughput_means):
        print(
            "ERREUR: le throughput_success_mean ne varie pas avec la taille du réseau."
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


def _read_nested_sizes(base_results_dir: Path, replications: list[int]) -> set[int]:
    sizes: set[int] = set()
    for size_dir in sorted(base_results_dir.glob("size_*")):
        if not size_dir.is_dir():
            continue
        try:
            size = int(size_dir.name.split("size_", 1)[1])
        except (IndexError, ValueError):
            continue
        rep_paths = [
            size_dir / f"rep_{replication}" / "aggregated_results.csv"
            for replication in replications
        ]
        if rep_paths and all(path.exists() for path in rep_paths):
            sizes.add(size)
    if not sizes:
        print(
            "Aucune taille complète détectée dans les sous-dossiers "
            f"{base_results_dir / 'size_<N>/rep_<R>'}."
        )
    return sizes


def _compose_post_simulation_report(
    per_size_stats: dict[int, dict[str, object]],
    per_size_diagnostics: dict[int, dict[str, float]],
) -> str:
    if not per_size_stats:
        return "Rapport post-simulation indisponible (aucune statistique collectée)."
    overall_success_sum = 0.0
    overall_success_count = 0
    overall_success_zero_count = 0
    overall_collision_sum = 0.0
    overall_collision_count = 0
    overall_collision_hist = _init_collision_histogram()
    overall_link_quality_sum = 0.0
    overall_link_quality_count = 0
    overall_link_quality_min: float | None = None
    overall_link_quality_max: float | None = None
    reward_zero_no_success = 0
    reward_zero_clipped = 0
    reward_zero_total = 0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_count = 0

    per_size_link_quality_mean: dict[int, float] = {}
    for size, stats in per_size_stats.items():
        success_sum = float(stats.get("success_sum", 0.0))
        success_count = int(stats.get("success_count", 0))
        success_zero_count = int(stats.get("success_zero_count", 0))
        collision_sum = float(stats.get("collision_sum", 0.0))
        collision_count = int(stats.get("collision_count", 0))
        link_quality_sum = float(stats.get("link_quality_sum", 0.0))
        link_quality_count = int(stats.get("link_quality_count", 0))
        link_quality_min = float(stats.get("link_quality_min", 0.0))
        link_quality_max = float(stats.get("link_quality_max", 0.0))
        overall_success_sum += success_sum
        overall_success_count += success_count
        overall_success_zero_count += success_zero_count
        overall_collision_sum += collision_sum
        overall_collision_count += collision_count
        for bucket, count in dict(stats.get("collision_hist", {})).items():
            overall_collision_hist[bucket] = overall_collision_hist.get(bucket, 0) + int(
                count
            )
        overall_link_quality_sum += link_quality_sum
        overall_link_quality_count += link_quality_count
        overall_link_quality_min = (
            link_quality_min
            if overall_link_quality_min is None
            else min(overall_link_quality_min, link_quality_min)
        )
        overall_link_quality_max = (
            link_quality_max
            if overall_link_quality_max is None
            else max(overall_link_quality_max, link_quality_max)
        )
        reward_zero_no_success += int(stats.get("reward_zero_no_success", 0))
        reward_zero_clipped += int(stats.get("reward_zero_clipped", 0))
        reward_zero_total += int(stats.get("reward_zero_total", 0))
        reward_min = (
            float(stats.get("reward_min", 0.0))
            if reward_min is None
            else min(reward_min, float(stats.get("reward_min", 0.0)))
        )
        reward_max = (
            float(stats.get("reward_max", 0.0))
            if reward_max is None
            else max(reward_max, float(stats.get("reward_max", 0.0)))
        )
        reward_count += int(stats.get("reward_count", 0))
        if link_quality_count > 0:
            per_size_link_quality_mean[size] = link_quality_sum / link_quality_count

    success_mean = (
        overall_success_sum / overall_success_count
        if overall_success_count > 0
        else 0.0
    )
    collision_mean = (
        overall_collision_sum / overall_collision_count
        if overall_collision_count > 0
        else 0.0
    )
    link_quality_mean = (
        overall_link_quality_sum / overall_link_quality_count
        if overall_link_quality_count > 0
        else 0.0
    )
    link_quality_min = 0.0 if overall_link_quality_min is None else overall_link_quality_min
    link_quality_max = 0.0 if overall_link_quality_max is None else overall_link_quality_max

    lines = [
        "Rapport post-simulation (étape 2)",
        "",
        "Taux de succès moyen:",
        f"- success_rate moyen global: {success_mean:.4f}",
    ]
    zero_success_ratio = (
        overall_success_zero_count / overall_success_count
        if overall_success_count > 0
        else 0.0
    )
    if overall_success_count > 0 and zero_success_ratio > 0.95:
        lines.extend(
            [
                "",
                (
                    "AVERTISSEMENT: plus de 95% des fenêtres ont un success_rate "
                    "nul. Vérifiez la configuration (trafic, SNIR, collisions)."
                ),
                "Simulation invalide : success_rate trop faible.",
            ]
        )
    if per_size_diagnostics:
        lines.append("- success_rate moyen par taille:")
        for size in sorted(per_size_diagnostics):
            metrics = per_size_diagnostics[size]
            lines.append(
                f"  - taille {size}: {metrics['success_mean']:.4f} "
                f"(min {metrics['success_min']:.4f}, max {metrics['success_max']:.4f})"
            )

    total_collisions = sum(overall_collision_hist.values()) or 1
    lines.extend(
        [
            "",
            "Distribution des collisions (collision_norm):",
            f"- moyenne globale: {collision_mean:.4f}",
        ]
    )
    for bucket in ("0-0.1", "0.1-0.3", "0.3-0.6", "0.6-1.0"):
        count = overall_collision_hist.get(bucket, 0)
        percent = 100.0 * count / total_collisions
        lines.append(f"  - {bucket}: {count} fenêtres ({percent:.1f}%)")

    lines.extend(
        [
            "",
            "Variation de link_quality:",
            (
                f"- moyenne globale: {link_quality_mean:.4f} "
                f"(min {link_quality_min:.4f}, max {link_quality_max:.4f})"
            ),
        ]
    )
    if per_size_link_quality_mean:
        lq_means = list(per_size_link_quality_mean.values())
        lq_delta = max(lq_means) - min(lq_means) if lq_means else 0.0
        variation_label = (
            "variation détectée"
            if lq_delta >= 1e-3
            else "variation très faible (quasi stable)"
        )
        lines.append(f"- amplitude inter-tailles: {lq_delta:.4f} ({variation_label})")
        lines.append("- moyenne link_quality par taille:")
        for size in sorted(per_size_link_quality_mean):
            lines.append(
                f"  - taille {size}: {per_size_link_quality_mean[size]:.4f}"
            )

    reward_zero_ratio = reward_zero_total / reward_count if reward_count > 0 else 0.0
    reward_min_value = 0.0 if reward_min is None else reward_min
    reward_max_value = 0.0 if reward_max is None else reward_max
    lines.extend(
        [
            "",
            "Analyse reward nul:",
            (
                f"- reward min/max observé: {reward_min_value:.4f}/"
                f"{reward_max_value:.4f}"
            ),
            f"- part de reward nul: {reward_zero_total}/{reward_count} "
            f"({reward_zero_ratio:.1%})",
        ]
    )
    if reward_zero_total > 0:
        no_success_ratio = reward_zero_no_success / reward_zero_total
        clipped_ratio = reward_zero_clipped / reward_zero_total
        lines.append(
            f"- reward nul sans succès: {reward_zero_no_success} "
            f"({no_success_ratio:.1%})"
        )
        lines.append(
            f"- reward nul malgré succès (>0): {reward_zero_clipped} "
            f"({clipped_ratio:.1%})"
        )
        if no_success_ratio > 0.6:
            conclusion = "Le reward nul provient majoritairement d'une absence de succès."
        elif clipped_ratio > 0.6:
            conclusion = (
                "Le reward nul provient majoritairement d'un écrêtage (pénalité collision)."
            )
        else:
            conclusion = (
                "Le reward nul est mixte: absence de succès et écrêtage contribuent."
            )
        lines.append(f"- conclusion: {conclusion}")
    else:
        lines.append("- conclusion: aucun reward nul détecté.")

    return "\n".join(lines)


def _assert_success_rate_threshold(
    per_size_stats: dict[int, dict[str, object]], threshold: float = 0.95
) -> None:
    overall_success_count = sum(
        int(stats.get("success_count", 0)) for stats in per_size_stats.values()
    )
    overall_success_zero_count = sum(
        int(stats.get("success_zero_count", 0)) for stats in per_size_stats.values()
    )
    if overall_success_count <= 0:
        return
    zero_ratio = overall_success_zero_count / overall_success_count
    if zero_ratio > threshold:
        raise RuntimeError("Simulation invalide : success_rate trop faible.")


def _write_post_simulation_report(
    output_dir: Path,
    per_size_stats: dict[int, dict[str, object]],
    per_size_diagnostics: dict[int, dict[str, float]],
) -> None:
    report = _compose_post_simulation_report(per_size_stats, per_size_diagnostics)
    report_path = output_dir / "post_simulation_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)


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

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_MULTI)
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
    place_legend(ax, legend_loc="above")
    output_plot_dir = output_dir / "plots"
    save_figure(fig, output_plot_dir, "summary_reward", use_tight=False)
    plt.close(fig)


def _simulate_density(
    task: tuple[int, int, list[int], dict[str, object], Path, Path | None, bool]
) -> dict[str, object]:
    (
        density,
        density_idx,
        replications,
        config,
        base_results_dir,
        timestamp_dir,
        flat_output,
    ) = task
    if config.get("debug_step2"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    raw_rows: list[dict[str, object]] = []
    per_rep_rows: dict[int, list[dict[str, object]]] = {
        replication: [] for replication in replications
    }
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
                lambda_collision=(
                    float(config["lambda_collision"])
                    if config.get("lambda_collision") is not None
                    else None
                ),
                traffic_coeff_min=float(config["traffic_coeff_min"]),
                traffic_coeff_max=float(config["traffic_coeff_max"]),
                traffic_coeff_enabled=bool(config["traffic_coeff_enabled"]),
                traffic_coeff_scale=float(config["traffic_coeff_scale"]),
                capture_probability=float(config["capture_probability"]),
                congestion_coeff=float(config["congestion_coeff"]),
                congestion_coeff_base=float(config["congestion_coeff_base"]),
                congestion_coeff_growth=float(config["congestion_coeff_growth"]),
                congestion_coeff_max=float(config["congestion_coeff_max"]),
                collision_size_factor=(
                    float(config["collision_size_factor"])
                    if config.get("collision_size_factor") is not None
                    else None
                ),
                traffic_coeff_clamp_min=float(config["traffic_coeff_clamp_min"]),
                traffic_coeff_clamp_max=float(config["traffic_coeff_clamp_max"]),
                traffic_coeff_clamp_enabled=bool(config["traffic_coeff_clamp_enabled"]),
                window_delay_enabled=bool(config["window_delay_enabled"]),
                window_delay_range_s=float(config["window_delay_range_s"]),
                reference_network_size=int(config["reference_network_size"]),
                reward_floor=(
                    float(config["reward_floor"])
                    if config.get("reward_floor") is not None
                    else None
                ),
                floor_on_zero_success=bool(config["floor_on_zero_success"]),
                debug_step2=bool(config.get("debug_step2", False)),
                reward_alert_level=str(config.get("reward_alert_level", "WARNING")),
            )
            for row in result.raw_rows:
                row["replication"] = replication
            raw_rows.extend(result.raw_rows)
            per_rep_rows[replication].extend(result.raw_rows)
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
    post_stats = _summarize_post_simulation(raw_rows)
    _log_size_diagnostics(int(density), diagnostics)

    if flat_output:
        write_simulation_results(base_results_dir, raw_rows, network_size=density)
        _log_results_written(base_results_dir, len(raw_rows))
        _assert_flat_output_files(base_results_dir, density)
        _log_unique_network_sizes(base_results_dir)
        if timestamp_dir is not None:
            write_simulation_results(timestamp_dir, raw_rows, network_size=density)
            _log_results_written(timestamp_dir, len(raw_rows))
            _assert_flat_output_files(timestamp_dir, density)
            _log_unique_network_sizes(timestamp_dir)
    else:
        for replication, rows in per_rep_rows.items():
            rep_dir = base_results_dir / f"size_{density}" / f"rep_{replication}"
            write_simulation_results(rep_dir, rows, network_size=density)
            _log_results_written(rep_dir, len(rows))
            _log_unique_network_sizes(rep_dir)
            if timestamp_dir is not None:
                rep_timestamp_dir = (
                    timestamp_dir / f"size_{density}" / f"rep_{replication}"
                )
                write_simulation_results(rep_timestamp_dir, rows, network_size=density)
                _log_results_written(rep_timestamp_dir, len(rows))
                _log_unique_network_sizes(rep_timestamp_dir)
    return {
        "density": density,
        "row_count": len(raw_rows),
        "selection_rows": selection_rows,
        "learning_curve_rows": learning_curve_rows,
        "diagnostics": diagnostics,
        "post_stats": post_stats,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    set_default_export_formats(export_formats)
    if args.debug_step2:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    base_seed = set_deterministic_seed(args.seeds_base)
    densities = parse_network_size_list(args.network_sizes)
    requested_sizes = list(densities)
    flat_output = bool(args.flat_output)
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
        timestamp_dir = base_results_dir / timestamp_tag(with_timezone=True)
        ensure_dir(timestamp_dir)
    aggregated_path = base_results_dir / "aggregated_results.csv"
    if flat_output and _is_non_empty_file(aggregated_path):
        size_bytes = aggregated_path.stat().st_size
        print(
            "aggregated_results.csv existant détecté "
            f"({size_bytes} octets) : aucune réinitialisation."
        )

    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    size_diagnostics: dict[int, dict[str, float]] = {}
    size_post_stats: dict[int, dict[str, object]] = {}
    total_rows = 0

    config: dict[str, object] = {
        "base_seed": base_seed,
        "traffic_mode": args.traffic_mode,
        "jitter_range_s": args.jitter_range_s,
        "window_duration_s": args.window_duration_s,
        "window_size": args.window_size,
        "lambda_collision": args.lambda_collision,
        "traffic_coeff_min": args.traffic_coeff_min,
        "traffic_coeff_max": args.traffic_coeff_max,
        "traffic_coeff_enabled": args.traffic_coeff_enabled,
        "traffic_coeff_scale": args.traffic_coeff_scale,
        "capture_probability": args.capture_probability,
        "congestion_coeff": args.congestion_coeff,
        "congestion_coeff_base": args.congestion_coeff_base,
        "congestion_coeff_growth": args.congestion_coeff_growth,
        "congestion_coeff_max": args.congestion_coeff_max,
        "collision_size_factor": args.collision_size_factor,
        "traffic_coeff_clamp_min": args.traffic_coeff_clamp_min,
        "traffic_coeff_clamp_max": args.traffic_coeff_clamp_max,
        "traffic_coeff_clamp_enabled": args.traffic_coeff_clamp_enabled,
        "window_delay_enabled": args.window_delay_enabled,
        "window_delay_range_s": args.window_delay_range_s,
        "reference_network_size": max(1, reference_network_size),
        "reward_floor": args.reward_floor,
        "floor_on_zero_success": args.floor_on_zero_success,
        "debug_step2": args.debug_step2,
        "reward_alert_level": args.reward_alert_level,
    }

    if flat_output:
        aggregated_sizes = _read_aggregated_sizes(
            base_results_dir / "aggregated_results.csv"
        )
    else:
        aggregated_sizes = _read_nested_sizes(base_results_dir, replications)
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
    if flat_output:
        print(f"Tailles déjà présentes dans aggregated_results.csv: {existing_label}")
    else:
        print(f"Tailles déjà présentes dans les sous-dossiers: {existing_label}")
    print(f"Tailles à simuler: {simulated_label}")

    tasks = [
        (
            density,
            density_idx,
            replications,
            config,
            base_results_dir,
            timestamp_dir,
            flat_output,
        )
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
            size_post_stats[int(result["density"])] = dict(result["post_stats"])
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_simulate_density, tasks):
                simulated_sizes.append(int(result["density"]))
                total_rows += int(result["row_count"])
                selection_rows.extend(result["selection_rows"])
                learning_curve_rows.extend(result["learning_curve_rows"])
                size_diagnostics[int(result["density"])] = dict(result["diagnostics"])
                size_post_stats[int(result["density"])] = dict(result["post_stats"])

    print(f"Rows written: {total_rows}")
    if flat_output and simulated_sizes:
        _assert_flat_output_sizes(base_results_dir, simulated_sizes)
    _verify_metric_variation(size_diagnostics)
    _write_post_simulation_report(base_results_dir, size_post_stats, size_diagnostics)
    _assert_success_rate_threshold(size_post_stats)

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

    if flat_output:
        aggregated_sizes = _read_aggregated_sizes(
            base_results_dir / "aggregated_results.csv"
        )
    else:
        aggregated_sizes = _read_nested_sizes(base_results_dir, replications)
    missing_sizes = sorted(set(requested_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        if flat_output:
            print(
                "ATTENTION: tailles manquantes dans aggregated_results.csv, "
                f"done.flag non écrit. Manquantes: {missing_label}"
            )
        else:
            print(
                "ATTENTION: tailles manquantes dans les sous-dossiers, "
                f"done.flag non écrit. Manquantes: {missing_label}"
            )
    else:
        (base_results_dir / "done.flag").write_text("done\n", encoding="utf-8")
        print("done.flag écrit (agrégation complète).")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")
    aggregated_path = base_results_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        if args.plot_summary:
            _plot_summary_reward(base_results_dir)
    elif args.plot_summary:
        print(
            "Plot de synthèse ignoré: aggregated_results.csv absent "
            "(utilisez --flat-output ou make_all_plots.py)."
        )


if __name__ == "__main__":
    main()
