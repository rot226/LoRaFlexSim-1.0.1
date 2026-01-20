"""Point d'entrée pour l'étape 1."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_simulation_results
from article_c.common.utils import (
    parse_network_size_list,
    replication_ids,
    set_deterministic_seed,
)
from article_c.step1.simulate_step1 import mixra_opt_budget_for_size, run_simulation

ALGORITHMS = ("adr", "mixra_h", "mixra_opt")
ALGORITHM_LABELS = {
    "adr": "ADR",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
}


def density_to_sent(density: float, base_sent: int = 120) -> int:
    """Convertit une taille de réseau en nombre de trames simulées."""
    return max(1, int(round(base_sent * density)))


def parse_snir_modes(value: str) -> list[str]:
    """Parse la liste des modes SNIR depuis une chaîne CSV."""
    return [item.strip() for item in value.split(",") if item.strip()]


def format_duration(seconds: float) -> str:
    """Formate une durée en HH:MM:SS."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def format_global_progress(*, percent: float, elapsed_s: float, eta_s: float) -> str:
    """Construit la ligne de progression globale pour l'étape 1."""
    elapsed_label = format_duration(elapsed_s)
    eta_label = format_duration(eta_s)
    return f"Progress global: {percent:.0f}% (elapsed {elapsed_label}, ETA {eta_label})"


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour l'étape 1."""
    parser = argparse.ArgumentParser(description="Exécute l'étape 1 de l'article C.")
    scenario_defaults = DEFAULT_CONFIG.scenario
    snir_defaults = DEFAULT_CONFIG.snir
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIG.scenario.network_sizes),
        help="Tailles de réseau (nombre de nœuds entiers, ex: 50 100 150).",
    )
    parser.add_argument(
        "--densities",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Alias de --network-sizes (déprécié).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Nombre de réplications par configuration.",
    )
    parser.add_argument(
        "--seeds_base",
        type=int,
        default=1000,
        help="Seed de base pour les réplications.",
    )
    parser.add_argument(
        "--seed",
        dest="seeds_base",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias de --seeds_base (déprécié).",
    )
    parser.add_argument(
        "--snir_modes",
        type=str,
        default="snir_on,snir_off",
        help="Liste des modes SNIR (ex: snir_on,snir_off).",
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default=scenario_defaults.traffic_mode,
        choices=("periodic", "poisson"),
        help="Modèle de trafic (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range",
        type=float,
        default=scenario_defaults.jitter_range,
        help="Amplitude du jitter (secondes).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=float(scenario_defaults.duration_s),
        help="Durée de la simulation (secondes).",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=snir_defaults.snir_threshold_db,
        help="Seuil SNIR (dB).",
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=snir_defaults.noise_floor_dbm,
        help="Bruit thermique (densité en dBm/Hz).",
    )
    parser.add_argument(
        "--mixra-opt-max-iterations",
        type=int,
        default=200,
        help="Nombre maximal d'itérations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-candidate-subset-size",
        type=int,
        default=200,
        help="Nombre maximal de nœuds optimisés par itération en MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-epsilon",
        type=float,
        default=1e-3,
        help="Seuil d'amélioration pour la convergence MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-max-evals",
        type=int,
        default=200,
        help="Nombre maximal d'évaluations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-budget",
        type=int,
        default=None,
        help=(
            "Budget cible d'évaluations pour MixRA-Opt (max d'évaluations). "
            "Si absent, un budget par taille est appliqué "
            "(ex: N=80→5000, N=160→10000, N=320→20000, N=640→40000, N=1280→80000)."
        ),
    )
    parser.add_argument(
        "--mixra-opt-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Active ou désactive MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-mode",
        choices=("fast", "balanced", "full"),
        default="balanced",
        help=(
            "Mode MixRA-Opt (balanced par défaut, fast pour un budget strict, "
            "full pour une optimisation plus longue sans fallback)."
        ),
    )
    parser.add_argument(
        "--mixra-opt-timeout",
        type=float,
        default=None,
        help=(
            "Timeout (secondes) pour MixRA-Opt afin d'éviter les blocages "
            "(None pour désactiver)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Répertoire de sortie des CSV.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche la progression des simulations.",
    )
    parser.add_argument(
        "--profile-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Affiche les durées des étapes (assignation SF, interférences, "
            "agrégation métriques) par taille de réseau."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Nombre de processus worker pour paralléliser les tailles.",
    )
    return parser


def _simulate_density(
    task: tuple[int, int, list[str], list[int], dict[str, object], Path, list[str]]
) -> dict[str, object]:
    density, density_idx, snir_modes, replications, config, output_dir, cluster_ids = task
    raw_rows: list[dict[str, object]] = []
    run_index = 0
    timing_totals = {"sf_assignment_s": 0.0, "interference_s": 0.0, "metrics_s": 0.0}
    timing_runs = 0
    runs_per_density = len(ALGORITHMS) * len(snir_modes) * len(replications)
    mixra_opt_budget = (
        config["mixra_opt_budget"]
        if config["mixra_opt_budget"] is not None
        else mixra_opt_budget_for_size(density)
    )
    for algo in ALGORITHMS:
        for snir_mode in snir_modes:
            for replication in replications:
                seed = int(config["seeds_base"]) + density_idx * runs_per_density + run_index
                run_index += 1
                set_deterministic_seed(seed)
                sent = density_to_sent(density)
                result = run_simulation(
                    sent=sent,
                    algorithm=algo,
                    seed=seed,
                    duration_s=float(config["duration_s"]),
                    traffic_mode=str(config["traffic_mode"]),
                    jitter_range_s=float(config["jitter_range"]),
                    mixra_opt_max_iterations=int(config["mixra_opt_max_iterations"]),
                    mixra_opt_candidate_subset_size=int(
                        config["mixra_opt_candidate_subset_size"]
                    ),
                    mixra_opt_epsilon=float(config["mixra_opt_epsilon"]),
                    mixra_opt_max_evaluations=int(config["mixra_opt_max_evals"]),
                    mixra_opt_budget=mixra_opt_budget,
                    mixra_opt_enabled=bool(config["mixra_opt_enabled"]),
                    mixra_opt_mode=str(config["mixra_opt_mode"]),
                    mixra_opt_timeout_s=config["mixra_opt_timeout"],
                    profile_timing=bool(config["profile_timing"]),
                )
                metrics_start = perf_counter() if config["profile_timing"] else 0.0
                cluster_stats = {
                    cluster: {"sent": 0, "received": 0} for cluster in cluster_ids
                }
                cluster_toa: dict[str, list[float]] = {cluster: [] for cluster in cluster_ids}
                for cluster, delivered in zip(
                    result.node_clusters, result.node_received, strict=True
                ):
                    cluster_stats[cluster]["sent"] += 1
                    if delivered:
                        cluster_stats[cluster]["received"] += 1
                for cluster, toa_s in zip(
                    result.node_clusters, result.toa_s_by_node, strict=True
                ):
                    cluster_toa[cluster].append(toa_s)
                for node_id, (packet_id, cluster, sf_selected) in enumerate(
                    zip(
                        result.packet_ids,
                        result.node_clusters,
                        result.sf_selected_by_node,
                        strict=True,
                    )
                ):
                    raw_rows.append(
                        {
                            "density": density,
                            "network_size": density,
                            "algo": algo,
                            "snir_mode": snir_mode,
                            "cluster": cluster,
                            "replication": replication,
                            "seed": seed,
                            "mixra_opt_fallback": result.mixra_opt_fallback,
                            "node_id": node_id,
                            "packet_id": packet_id,
                            "sf_selected": sf_selected,
                        }
                    )
                for cluster, stats in cluster_stats.items():
                    sent_cluster = stats["sent"]
                    received_cluster = stats["received"]
                    pdr_cluster = received_cluster / sent_cluster if sent_cluster > 0 else 0.0
                    mean_toa_s = mean(cluster_toa[cluster]) if cluster_toa[cluster] else 0.0
                    raw_rows.append(
                        {
                            "density": density,
                            "network_size": density,
                            "algo": algo,
                            "snir_mode": snir_mode,
                            "cluster": cluster,
                            "replication": replication,
                            "seed": seed,
                            "mixra_opt_fallback": result.mixra_opt_fallback,
                            "sent": sent_cluster,
                            "received": received_cluster,
                            "pdr": pdr_cluster,
                            "mean_toa_s": mean_toa_s,
                        }
                    )
                raw_rows.append(
                    {
                        "density": density,
                        "network_size": density,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": "all",
                        "replication": replication,
                        "seed": seed,
                        "mixra_opt_fallback": result.mixra_opt_fallback,
                        "sent": result.sent,
                        "received": result.received,
                        "pdr": result.pdr,
                        "mean_toa_s": result.mean_toa_s,
                    }
                )
                if config["profile_timing"] and result.timing_s is not None:
                    timing_totals["sf_assignment_s"] += result.timing_s.get(
                        "sf_assignment_s", 0.0
                    )
                    timing_totals["interference_s"] += result.timing_s.get(
                        "interference_s", 0.0
                    )
                    timing_totals["metrics_s"] += perf_counter() - metrics_start
                    timing_runs += 1
    write_simulation_results(output_dir, raw_rows)
    timing_summary = None
    if config["profile_timing"] and timing_runs > 0:
        mean_assignment = timing_totals["sf_assignment_s"] / timing_runs
        mean_interference = timing_totals["interference_s"] / timing_runs
        mean_metrics = timing_totals["metrics_s"] / timing_runs
        timing_summary = (
            "Profiling taille réseau "
            f"{density}: assignation SF {mean_assignment:.6f}s, "
            f"interférences {mean_interference:.6f}s, "
            f"agrégation métriques {mean_metrics:.6f}s "
            f"(moyenne sur {timing_runs} runs)."
        )
    return {
        "density": density,
        "row_count": len(raw_rows),
        "timing_summary": timing_summary,
        "run_count": runs_per_density,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    densities = parse_network_size_list(args.network_sizes)
    snir_modes = parse_snir_modes(args.snir_modes)
    replications = replication_ids(args.replications)
    output_dir = Path(args.outdir)
    simulated_sizes: list[int] = []

    total_runs = len(densities) * len(ALGORITHMS) * len(snir_modes) * len(replications)
    completed_runs = 0
    total_rows = 0
    rows_per_size: Counter[int] = Counter()
    progress_start = perf_counter()
    cluster_ids = list(DEFAULT_CONFIG.qos.clusters)

    config: dict[str, object] = {
        "seeds_base": args.seeds_base,
        "duration_s": args.duration_s,
        "traffic_mode": args.traffic_mode,
        "jitter_range": args.jitter_range,
        "mixra_opt_max_iterations": args.mixra_opt_max_iterations,
        "mixra_opt_candidate_subset_size": args.mixra_opt_candidate_subset_size,
        "mixra_opt_epsilon": args.mixra_opt_epsilon,
        "mixra_opt_max_evals": args.mixra_opt_max_evals,
        "mixra_opt_budget": args.mixra_opt_budget,
        "mixra_opt_enabled": args.mixra_opt_enabled,
        "mixra_opt_mode": args.mixra_opt_mode,
        "mixra_opt_timeout": args.mixra_opt_timeout,
        "profile_timing": args.profile_timing,
    }

    tasks = [
        (density, density_idx, snir_modes, replications, config, output_dir, cluster_ids)
        for density_idx, density in enumerate(densities)
    ]

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = map(_simulate_density, tasks)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            results = pool.imap_unordered(_simulate_density, tasks)
            for result in results:
                simulated_sizes.append(int(result["density"]))
                total_rows += int(result["row_count"])
                rows_per_size[int(result["density"])] += int(result["row_count"])
                completed_runs += int(result["run_count"])
                if args.progress and total_runs > 0:
                    percent = (completed_runs / total_runs) * 100
                    elapsed_s = perf_counter() - progress_start
                    eta_s = (
                        (elapsed_s / completed_runs) * (total_runs - completed_runs)
                        if completed_runs > 0
                        else 0.0
                    )
                    print(
                        format_global_progress(
                            percent=percent, elapsed_s=elapsed_s, eta_s=eta_s
                        )
                    )
                if result.get("timing_summary"):
                    print(result["timing_summary"])
            results = None
    if worker_count == 1:
        for result in results:
            simulated_sizes.append(int(result["density"]))
            total_rows += int(result["row_count"])
            rows_per_size[int(result["density"])] += int(result["row_count"])
            completed_runs += int(result["run_count"])
            if args.progress and total_runs > 0:
                percent = (completed_runs / total_runs) * 100
                elapsed_s = perf_counter() - progress_start
                eta_s = (
                    (elapsed_s / completed_runs) * (total_runs - completed_runs)
                    if completed_runs > 0
                    else 0.0
                )
                print(
                    format_global_progress(
                        percent=percent, elapsed_s=elapsed_s, eta_s=eta_s
                    )
                )
            if result.get("timing_summary"):
                print(result["timing_summary"])

    print(f"Rows written: {total_rows}")
    if rows_per_size:
        sizes_summary = ", ".join(
            f"{size}={count}" for size, count in sorted(rows_per_size.items())
        )
        print(f"Rows per size: {sizes_summary}")
    (output_dir / "done.flag").write_text("done\n", encoding="utf-8")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")


if __name__ == "__main__":
    main()
