"""Point d'entrée pour l'étape 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_simulation_results
from article_c.common.utils import (
    parse_network_size_list,
    replication_ids,
    set_deterministic_seed,
)
from article_c.step1.simulate_step1 import run_simulation

ALGORITHMS = ("adr", "mixra_h", "mixra_opt")


def density_to_sent(density: float, base_sent: int = 120) -> int:
    """Convertit une taille de réseau en nombre de trames simulées."""
    return max(1, int(round(base_sent * density)))


def parse_snir_modes(value: str) -> list[str]:
    """Parse la liste des modes SNIR depuis une chaîne CSV."""
    return [item.strip() for item in value.split(",") if item.strip()]


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
        "--mixra-opt-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Active ou désactive MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-mode",
        choices=("fast", "full"),
        default="fast",
        help="Mode MixRA-Opt (fast par défaut, full pour une optimisation plus longue).",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    densities = parse_network_size_list(args.network_sizes)
    snir_modes = parse_snir_modes(args.snir_modes)
    replications = replication_ids(args.replications)
    output_dir = Path(args.outdir)

    raw_rows: list[dict[str, object]] = []
    run_index = 0
    total_runs = len(densities) * len(ALGORITHMS) * len(snir_modes) * len(replications)
    completed_runs = 0
    cluster_ids = list(DEFAULT_CONFIG.qos.clusters)
    for density in densities:
        for algo in ALGORITHMS:
            for snir_mode in snir_modes:
                for replication in replications:
                    seed = args.seeds_base + run_index
                    run_index += 1
                    set_deterministic_seed(seed)
                    sent = density_to_sent(density)
                    result = run_simulation(
                        sent=sent,
                        algorithm=algo,
                        seed=seed,
                        duration_s=args.duration_s,
                        traffic_mode=args.traffic_mode,
                        jitter_range_s=args.jitter_range,
                        mixra_opt_max_iterations=args.mixra_opt_max_iterations,
                        mixra_opt_candidate_subset_size=args.mixra_opt_candidate_subset_size,
                        mixra_opt_epsilon=args.mixra_opt_epsilon,
                        mixra_opt_max_evaluations=args.mixra_opt_max_evals,
                        mixra_opt_enabled=args.mixra_opt_enabled,
                        mixra_opt_mode=args.mixra_opt_mode,
                    )
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
                    for node_id, (cluster, sf_selected) in enumerate(
                        zip(result.node_clusters, result.sf_selected_by_node, strict=True)
                    ):
                        raw_rows.append(
                            {
                                "density": density,
                                "algo": algo,
                                "snir_mode": snir_mode,
                                "cluster": cluster,
                                "replication": replication,
                                "seed": seed,
                                "node_id": node_id,
                                "packet_id": node_id,
                                "sf_selected": sf_selected,
                            }
                        )
                    for cluster, stats in cluster_stats.items():
                        sent_cluster = stats["sent"]
                        received_cluster = stats["received"]
                        pdr_cluster = (
                            received_cluster / sent_cluster if sent_cluster > 0 else 0.0
                        )
                        mean_toa_s = mean(cluster_toa[cluster]) if cluster_toa[cluster] else 0.0
                        raw_rows.append(
                            {
                                "density": density,
                                "algo": algo,
                                "snir_mode": snir_mode,
                                "cluster": cluster,
                                "replication": replication,
                                "seed": seed,
                                "sent": sent_cluster,
                                "received": received_cluster,
                                "pdr": pdr_cluster,
                                "mean_toa_s": mean_toa_s,
                            }
                        )
                    raw_rows.append(
                        {
                            "density": density,
                            "algo": algo,
                            "snir_mode": snir_mode,
                            "cluster": "all",
                            "replication": replication,
                            "seed": seed,
                            "sent": result.sent,
                            "received": result.received,
                            "pdr": result.pdr,
                            "mean_toa_s": result.mean_toa_s,
                        }
                    )
                    completed_runs += 1
                    if args.progress and total_runs > 0:
                        percent = (completed_runs / total_runs) * 100
                        print(
                            f"Progress: {completed_runs}/{total_runs} ({percent:.1f}%)"
                        )

    write_simulation_results(output_dir, raw_rows)


if __name__ == "__main__":
    main()
