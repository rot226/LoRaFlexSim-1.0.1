"""Point d'entrée pour l'étape 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_simulation_results
from article_c.common.utils import parse_density_list, replication_ids, set_deterministic_seed
from article_c.step1.simulate_step1 import run_simulation

ALGORITHMS = ("adr", "mixra_h", "mixra_opt")


def density_to_sent(density: float, base_sent: int = 120) -> int:
    """Convertit une densité en nombre de trames simulées."""
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
        "--densities",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Liste des densités (ex: 0.1 0.5 1.0).",
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
        help="Seuil SNIR/capture (dB).",
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
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Répertoire de sortie des CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    densities = parse_density_list(args.densities)
    snir_modes = parse_snir_modes(args.snir_modes)
    replications = replication_ids(args.replications)
    output_dir = Path(args.outdir)

    raw_rows: list[dict[str, object]] = []
    run_index = 0
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
                    )
                    cluster_stats = {cluster: {"sent": 0, "received": 0} for cluster in cluster_ids}
                    for cluster, delivered in zip(
                        result.node_clusters, result.node_received, strict=True
                    ):
                        cluster_stats[cluster]["sent"] += 1
                        if delivered:
                            cluster_stats[cluster]["received"] += 1
                    for cluster, stats in cluster_stats.items():
                        sent_cluster = stats["sent"]
                        received_cluster = stats["received"]
                        pdr_cluster = (
                            received_cluster / sent_cluster if sent_cluster > 0 else 0.0
                        )
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
                        }
                    )

    write_simulation_results(output_dir, raw_rows)


if __name__ == "__main__":
    main()
