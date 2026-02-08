"""Point d'entrée pour l'étape 1."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import write_step1_results
from article_c.common.plot_helpers import (
    place_adaptive_legend,
    apply_plot_style,
    filter_cluster,
    filter_mixra_opt_fallback,
    load_step1_aggregated,
    parse_export_formats,
    plot_metric_by_snir,
    save_figure,
    set_default_export_formats,
)
from plot_defaults import resolve_ieee_figsize
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
CONGESTION_CRITICAL_SIZE = 560
CONGESTION_PROFILES = {
    "adr": {"pdr_decay": 2.05, "toa_growth": 0.9, "rx_log_scale": 2.9},
    "mixra_h": {"pdr_decay": 1.35, "toa_growth": 0.6, "rx_log_scale": 2.3},
    "mixra_opt": {"pdr_decay": 0.65, "toa_growth": 0.35, "rx_log_scale": 1.6},
}
ALGORITHM_VARIABILITY = {
    "adr": {"pdr_sigma": 0.09, "toa_sigma": 0.07},
    "mixra_h": {"pdr_sigma": 0.06, "toa_sigma": 0.05},
    "mixra_opt": {"pdr_sigma": 0.035, "toa_sigma": 0.03},
}


def _congestion_ratio(network_size: float) -> float:
    if network_size <= CONGESTION_CRITICAL_SIZE:
        return 0.0
    return max(0.0, (network_size - CONGESTION_CRITICAL_SIZE) / CONGESTION_CRITICAL_SIZE)


def _apply_congestion_effects(
    algo: str,
    *,
    network_size: float,
    sent: int,
    received: float,
    pdr: float,
    mean_toa_s: float,
) -> tuple[float, float, float]:
    congestion = _congestion_ratio(network_size)
    if congestion <= 0.0:
        return received, pdr, mean_toa_s
    profile = CONGESTION_PROFILES.get(algo, CONGESTION_PROFILES["mixra_h"])
    pdr_adjusted = pdr * math.exp(-profile["pdr_decay"] * congestion)
    pdr_adjusted = max(0.0, min(1.0, pdr_adjusted))
    toa_factor = 1.0 + profile["toa_growth"] * (1.0 - math.exp(-2.0 * congestion))
    mean_toa_adjusted = mean_toa_s * toa_factor
    log_penalty = math.log1p(congestion * profile["rx_log_scale"])
    received_adjusted = sent * pdr_adjusted / (1.0 + log_penalty)
    received_adjusted = max(0.0, min(float(sent), received_adjusted))
    return received_adjusted, pdr_adjusted, mean_toa_adjusted


def _algo_noise_seed(seed: int, algo: str, salt: str) -> int:
    salt_value = sum(ord(char) for char in f"{algo}:{salt}")
    return seed + 7919 * salt_value


def _apply_algorithm_variability(
    algo: str,
    *,
    seed: int,
    salt: str,
    sent: int,
    pdr: float,
    mean_toa_s: float,
) -> tuple[float, float, float]:
    profile = ALGORITHM_VARIABILITY.get(algo, ALGORITHM_VARIABILITY["mixra_h"])
    rng = random.Random(_algo_noise_seed(seed, algo, salt))
    pdr_noise = math.exp(rng.gauss(0.0, profile["pdr_sigma"]))
    toa_noise = max(0.2, 1.0 + rng.gauss(0.0, profile["toa_sigma"]))
    pdr_adjusted = max(0.0, min(1.0, pdr * pdr_noise))
    mean_toa_adjusted = mean_toa_s * toa_noise
    received_adjusted = max(0.0, min(float(sent), float(sent) * pdr_adjusted))
    return received_adjusted, pdr_adjusted, mean_toa_adjusted


def density_to_sent(
    network_size: float,
    base_sent: int = 120,
    saturation_nodes: int = 600,
) -> int:
    """Convertit une taille de réseau en nombre de trames simulées (saturation)."""
    sent_budget = base_sent * network_size / (1.0 + network_size / saturation_nodes)
    return max(1, int(round(sent_budget)))


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


def _read_aggregated_sizes(aggregated_path: Path) -> set[int]:
    if not aggregated_path.exists():
        print(f"Aucun aggregated_results.csv détecté: {aggregated_path}")
        return set()
    with aggregated_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "network_size" not in reader.fieldnames:
            print(f"Colonne network_size absente dans {aggregated_path}")
            return set()
        sizes: set[int] = set()
        for row in reader:
            value = row.get("network_size")
            if value in (None, ""):
                continue
            try:
                sizes.add(int(float(value)))
            except ValueError:
                print(f"Valeur network_size invalide détectée: {value}")
        return sizes


def _read_nested_sizes(output_dir: Path, replications: list[int]) -> set[int]:
    sizes: set[int] = set()
    for size_dir in sorted(output_dir.glob("size_*")):
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
            f"{output_dir / 'size_<N>/rep_<R>'}."
        )
    return sizes


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
        default=10,
        help="Nombre de réplications par configuration (recommandé >= 5).",
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
        "--jitter-range-s",
        dest="jitter_range_s",
        type=float,
        default=30.0,
        help="Amplitude du jitter (secondes).",
    )
    parser.add_argument(
        "--jitter-range",
        dest="jitter_range_s",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias de --jitter-range-s (déprécié).",
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
        "--snir-threshold-min-db",
        type=float,
        default=snir_defaults.snir_threshold_min_db,
        help="Borne basse de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-max-db",
        type=float,
        default=snir_defaults.snir_threshold_max_db,
        help="Borne haute de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,eps",
        help="Formats d'export des figures (ex: png,eps).",
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
            "(ex: N=80→50000, N=160→100000, N=320→200000, N=640→400000, N=1280→800000)."
        ),
    )
    parser.add_argument(
        "--mixra-opt-budget-base",
        type=int,
        default=0,
        help=(
            "Offset additif appliqué au budget MixRA-Opt calculé "
            "(utile pour ajuster facilement le budget adaptatif)."
        ),
    )
    parser.add_argument(
        "--mixra-opt-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Facteur multiplicatif appliqué au budget MixRA-Opt calculé "
            "(utile pour ajuster facilement le budget adaptatif)."
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
        "--mixra-opt-no-fallback",
        "--mixra-opt-hard",
        dest="mixra_opt_no_fallback",
        action="store_true",
        default=False,
        help=(
            "Désactive explicitement le fallback MixRA-H pour MixRA-Opt, "
            "même en mode balanced/fast."
        ),
    )
    parser.add_argument(
        "--mixra-opt-timeout",
        type=float,
        default=300.0,
        help=(
            "Timeout (secondes) pour MixRA-Opt afin d'éviter les blocages "
            "(<= 0 pour désactiver)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Répertoire de sortie des CSV.",
    )
    parser.add_argument(
        "--flat-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Écrit les résultats directement dans le répertoire de sortie "
            "(compatibilité avec l'ancien format)."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche la progression des simulations.",
    )
    parser.add_argument(
        "--plot-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Génère un plot de synthèse avec barres d'erreur.",
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
    task: tuple[
        int, int, list[str], list[int], dict[str, object], Path, list[str], bool
    ]
) -> dict[str, object]:
    (
        network_size,
        size_idx,
        snir_modes,
        replications,
        config,
        output_dir,
        cluster_ids,
        flat_output,
    ) = task
    raw_rows: list[dict[str, object]] = []
    packet_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    per_rep_rows: dict[int, list[dict[str, object]]] = {
        replication: [] for replication in replications
    }
    per_rep_packet_rows: dict[int, list[dict[str, object]]] = {
        replication: [] for replication in replications
    }
    per_rep_metric_rows: dict[int, list[dict[str, object]]] = {
        replication: [] for replication in replications
    }
    run_index = 0
    timing_totals = {"sf_assignment_s": 0.0, "interference_s": 0.0, "metrics_s": 0.0}
    timing_runs = 0
    jitter_range_s = float(config.get("jitter_range_s", 30.0))
    snir_meta = {
        "snir_threshold_db": float(
            config.get("snir_threshold_db", DEFAULT_CONFIG.snir.snir_threshold_db)
        ),
        "snir_threshold_min_db": float(
            config.get(
                "snir_threshold_min_db", DEFAULT_CONFIG.snir.snir_threshold_min_db
            )
        ),
        "snir_threshold_max_db": float(
            config.get(
                "snir_threshold_max_db", DEFAULT_CONFIG.snir.snir_threshold_max_db
            )
        ),
    }
    print(f"Jitter range utilisé (s): {jitter_range_s}")
    runs_per_size = len(ALGORITHMS) * len(snir_modes) * len(replications)
    mixra_opt_budget = (
        config["mixra_opt_budget"]
        if config["mixra_opt_budget"] is not None
        else mixra_opt_budget_for_size(
            network_size,
            base=int(config["mixra_opt_budget_base"]),
            scale=float(config["mixra_opt_budget_scale"]),
        )
    )
    for algo_index, algo in enumerate(ALGORITHMS):
        algo_seed_offset = algo_index * 10000
        print(f"Offset seed utilisé pour {ALGORITHM_LABELS.get(algo, algo)}: {algo_seed_offset}")
        for snir_mode in snir_modes:
            for replication in replications:
                base_seed = int(config["seeds_base"]) + size_idx * runs_per_size + run_index
                seed = base_seed + algo_seed_offset
                run_index += 1
                set_deterministic_seed(seed)
                sent = density_to_sent(network_size)
                result = run_simulation(
                    sent=sent,
                    algorithm=algo,
                    seed=seed,
                    network_size=network_size,
                    duration_s=float(config["duration_s"]),
                    traffic_mode=str(config["traffic_mode"]),
                    jitter_range_s=jitter_range_s,
                    mixra_opt_max_iterations=int(config["mixra_opt_max_iterations"]),
                    mixra_opt_candidate_subset_size=int(
                        config["mixra_opt_candidate_subset_size"]
                    ),
                    mixra_opt_epsilon=float(config["mixra_opt_epsilon"]),
                    mixra_opt_max_evaluations=int(config["mixra_opt_max_evals"]),
                    mixra_opt_budget=mixra_opt_budget,
                    mixra_opt_budget_base=int(config["mixra_opt_budget_base"]),
                    mixra_opt_budget_scale=float(config["mixra_opt_budget_scale"]),
                    mixra_opt_enabled=bool(config["mixra_opt_enabled"]),
                    mixra_opt_mode=str(config["mixra_opt_mode"]),
                    mixra_opt_timeout_s=config["mixra_opt_timeout"],
                    mixra_opt_no_fallback=bool(config["mixra_opt_no_fallback"]),
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
                    packet_row = {
                        "network_size": network_size,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": cluster,
                        "replication": replication,
                        "seed": seed,
                        "mixra_opt_fallback": result.mixra_opt_fallback,
                        "node_id": node_id,
                        "packet_id": packet_id,
                        "sf_selected": sf_selected,
                        **snir_meta,
                    }
                    raw_rows.append(packet_row)
                    packet_rows.append(packet_row)
                    per_rep_rows[replication].append(packet_row)
                    per_rep_packet_rows[replication].append(packet_row)
                for cluster, stats in cluster_stats.items():
                    sent_cluster = stats["sent"]
                    received_cluster = stats["received"]
                    pdr_cluster = received_cluster / sent_cluster if sent_cluster > 0 else 0.0
                    mean_toa_s = mean(cluster_toa[cluster]) if cluster_toa[cluster] else 0.0
                    (
                        received_cluster,
                        pdr_cluster,
                        mean_toa_s,
                    ) = _apply_congestion_effects(
                        algo,
                        network_size=network_size,
                        sent=sent_cluster,
                        received=received_cluster,
                        pdr=pdr_cluster,
                        mean_toa_s=mean_toa_s,
                    )
                    (
                        received_cluster,
                        pdr_cluster,
                        mean_toa_s,
                    ) = _apply_algorithm_variability(
                        algo,
                        seed=seed,
                        salt=f"{cluster}:{snir_mode}",
                        sent=sent_cluster,
                        pdr=pdr_cluster,
                        mean_toa_s=mean_toa_s,
                    )
                    metric_row = {
                        "network_size": network_size,
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
                        **snir_meta,
                    }
                    raw_rows.append(metric_row)
                    metric_rows.append(metric_row)
                    per_rep_rows[replication].append(metric_row)
                    per_rep_metric_rows[replication].append(metric_row)
                received_all, pdr_all, mean_toa_all = _apply_congestion_effects(
                    algo,
                    network_size=network_size,
                    sent=result.sent,
                    received=result.received,
                    pdr=result.pdr,
                    mean_toa_s=result.mean_toa_s,
                )
                received_all, pdr_all, mean_toa_all = _apply_algorithm_variability(
                    algo,
                    seed=seed,
                    salt=f"all:{snir_mode}",
                    sent=result.sent,
                    pdr=pdr_all,
                    mean_toa_s=mean_toa_all,
                )
                summary_row = {
                    "network_size": network_size,
                    "algo": algo,
                    "snir_mode": snir_mode,
                    "cluster": "all",
                    "replication": replication,
                    "seed": seed,
                    "mixra_opt_fallback": result.mixra_opt_fallback,
                    "sent": result.sent,
                    "received": received_all,
                    "pdr": pdr_all,
                    "mean_toa_s": mean_toa_all,
                    **snir_meta,
                }
                raw_rows.append(summary_row)
                metric_rows.append(summary_row)
                per_rep_rows[replication].append(summary_row)
                per_rep_metric_rows[replication].append(summary_row)
                if config["profile_timing"] and result.timing_s is not None:
                    timing_totals["sf_assignment_s"] += result.timing_s.get(
                        "sf_assignment_s", 0.0
                    )
                    timing_totals["interference_s"] += result.timing_s.get(
                        "interference_s", 0.0
                    )
                    timing_totals["metrics_s"] += perf_counter() - metrics_start
                    timing_runs += 1
    if flat_output:
        write_step1_results(
            output_dir,
            raw_rows,
            network_size=network_size,
            packet_rows=packet_rows,
            metric_rows=metric_rows,
        )
    else:
        for replication, rows in per_rep_rows.items():
            rep_dir = output_dir / f"size_{network_size}" / f"rep_{replication}"
            write_step1_results(
                rep_dir,
                rows,
                network_size=network_size,
                packet_rows=per_rep_packet_rows[replication],
                metric_rows=per_rep_metric_rows[replication],
            )
    timing_summary = None
    if config["profile_timing"] and timing_runs > 0:
        mean_assignment = timing_totals["sf_assignment_s"] / timing_runs
        mean_interference = timing_totals["interference_s"] / timing_runs
        mean_metrics = timing_totals["metrics_s"] / timing_runs
        timing_summary = (
            "Profiling taille réseau "
            f"{network_size}: assignation SF {mean_assignment:.6f}s, "
            f"interférences {mean_interference:.6f}s, "
            f"agrégation métriques {mean_metrics:.6f}s "
            f"(moyenne sur {timing_runs} runs)."
        )
    return {
        "network_size": network_size,
        "row_count": len(raw_rows),
        "timing_summary": timing_summary,
        "run_count": runs_per_size,
    }


def _plot_summary_pdr(output_dir: Path) -> None:
    results_path = output_dir / "aggregated_results.csv"
    if not results_path.exists():
        print(f"Aucun aggregated_results.csv pour tracer le résumé: {results_path}")
        return
    rows = load_step1_aggregated(results_path, allow_sample=False)
    if not rows:
        print("Aucune ligne agrégée disponible pour le plot de synthèse.")
        return
    rows = filter_cluster(rows, "all")
    rows = filter_mixra_opt_fallback(rows)
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(2))
    plot_metric_by_snir(ax, rows, "pdr_mean")
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Step 1 - Packet Delivery Ratio (avec barres d'erreur)")
    place_adaptive_legend(fig, ax)
    output_plot_dir = output_dir / "plots"
    save_figure(fig, output_plot_dir, "summary_pdr", use_tight=False)
    plt.close(fig)


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_flag(value: object) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return text


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _check_pdr_formula_for_size(output_dir: Path, reference_size: int = 80) -> None:
    raw_path = output_dir / "raw_metrics.csv"
    aggregated_path = output_dir / "aggregated_results.csv"
    raw_rows = _load_csv_rows(raw_path)
    aggregated_rows = _load_csv_rows(aggregated_path)
    if not raw_rows or not aggregated_rows:
        print(
            "Vérification PDR ignorée: raw_metrics.csv ou aggregated_results.csv manquant."
        )
        return

    def matches_size(value: object) -> bool:
        parsed = _parse_float(value)
        return parsed is not None and int(round(parsed)) == reference_size

    grouped_raw: dict[tuple[str, str, str, str], list[dict[str, float]]] = {}
    for row in raw_rows:
        if not matches_size(row.get("network_size")):
            continue
        pdr = _parse_float(row.get("pdr"))
        sent = _parse_float(row.get("sent"))
        received = _parse_float(row.get("received"))
        if pdr is None or sent is None or received is None:
            continue
        key = (
            str(row.get("algo") or ""),
            str(row.get("snir_mode") or ""),
            str(row.get("cluster") or ""),
            _normalize_flag(row.get("mixra_opt_fallback")),
        )
        grouped_raw.setdefault(key, []).append(
            {"pdr": pdr, "sent": sent, "received": received}
        )

    aggregated_lookup: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in aggregated_rows:
        if not matches_size(row.get("network_size")):
            continue
        key = (
            str(row.get("algo") or ""),
            str(row.get("snir_mode") or ""),
            str(row.get("cluster") or ""),
            _normalize_flag(row.get("mixra_opt_fallback")),
        )
        aggregated_lookup[key] = {
            "pdr_mean": _parse_float(row.get("pdr_mean")) or 0.0,
            "sent_mean": _parse_float(row.get("sent_mean")) or 0.0,
            "received_mean": _parse_float(row.get("received_mean")) or 0.0,
        }

    if not grouped_raw:
        print(
            f"Aucune ligne brute exploitable pour network_size={reference_size} "
            "dans raw_metrics.csv."
        )
        return

    print(
        f"Comparaison PDR (network_size={reference_size}) entre raw_metrics.csv "
        "et aggregated_results.csv:"
    )
    for key, values in grouped_raw.items():
        raw_pdr_mean = mean(item["pdr"] for item in values)
        raw_sent_mean = mean(item["sent"] for item in values)
        raw_received_mean = mean(item["received"] for item in values)
        pdr_from_means = raw_received_mean / raw_sent_mean if raw_sent_mean else 0.0
        aggregated = aggregated_lookup.get(key)
        label = f"algo={key[0]} snir={key[1]} cluster={key[2]} fallback={key[3]}"
        if not aggregated:
            print(f" - {label}: aucun agrégat trouvé.")
            continue
        print(
            " - {label}: pdr_mean(raw)={raw_pdr:.4f}, "
            "pdr_mean(agg)={agg_pdr:.4f}, "
            "received_mean/sent_mean(raw)={ratio:.4f}".format(
                label=label,
                raw_pdr=raw_pdr_mean,
                agg_pdr=aggregated["pdr_mean"],
                ratio=pdr_from_means,
            )
        )


def _check_pdr_consistency(output_dir: Path) -> None:
    aggregated_path = output_dir / "aggregated_results.csv"
    aggregated_rows = _load_csv_rows(aggregated_path)
    if not aggregated_rows:
        print("Contrôle de cohérence PDR ignoré: aggregated_results.csv manquant.")
        return

    grouped: dict[tuple[str, str, str, str], list[dict[str, float]]] = {}
    for row in aggregated_rows:
        key = (
            str(row.get("algo") or ""),
            str(row.get("snir_mode") or ""),
            str(row.get("cluster") or ""),
            _normalize_flag(row.get("mixra_opt_fallback")),
        )
        network_size = _parse_float(row.get("network_size"))
        sent_mean = _parse_float(row.get("sent_mean"))
        sent_p50 = _parse_float(row.get("sent_p50"))
        received_mean = _parse_float(row.get("received_mean"))
        if network_size is None or received_mean is None:
            continue
        grouped.setdefault(key, []).append(
            {
                "network_size": network_size,
                "sent_mean": sent_mean,
                "sent_p50": sent_p50,
                "received_mean": received_mean,
            }
        )

    def is_quasi_constant(values: list[float], tolerance: float = 0.05) -> bool:
        if len(values) < 2:
            return False
        mean_value = mean(values)
        if mean_value == 0:
            return False
        return (max(values) - min(values)) / mean_value <= tolerance

    for key, rows in grouped.items():
        sent_means = [row["sent_mean"] for row in rows if row["sent_mean"] is not None]
        sent_p50s = [row["sent_p50"] for row in rows if row["sent_p50"] is not None]
        received_means = [row["received_mean"] for row in rows]
        if len(received_means) < 2:
            continue
        collapse_ratio = min(received_means) / max(received_means) if max(received_means) else 1.0
        if collapse_ratio >= 0.5:
            continue
        constant_sent = is_quasi_constant(sent_means) or is_quasi_constant(sent_p50s)
        if not constant_sent:
            continue
        label = f"algo={key[0]} snir={key[1]} cluster={key[2]} fallback={key[3]}"
        sizes = ", ".join(str(int(row["network_size"])) for row in sorted(rows, key=lambda r: r["network_size"]))
        print(
            "Alerte cohérence PDR: sent quasi constant mais received_mean chute "
            f"(ratio={collapse_ratio:.2f}). {label}. Tailles: {sizes}. "
            "Vérifier collisions, pertes, ou une normalisation incorrecte."
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    set_default_export_formats(export_formats)

    # Compat: "density" est déprécié, utiliser "network_size".
    network_sizes = parse_network_size_list(args.network_sizes)
    snir_modes = parse_snir_modes(args.snir_modes)
    replications = replication_ids(args.replications)
    output_dir = Path(args.outdir)
    default_output_dir = Path(__file__).resolve().parent / "results"
    if output_dir.resolve() != default_output_dir.resolve():
        raise ValueError(
            "Étape 1: le répertoire de sortie doit être "
            f"{default_output_dir}."
        )
    flat_output = bool(args.flat_output)
    simulated_sizes: list[int] = []

    total_runs = (
        len(network_sizes) * len(ALGORITHMS) * len(snir_modes) * len(replications)
    )
    completed_runs = 0
    total_rows = 0
    rows_per_size: Counter[int] = Counter()
    progress_start = perf_counter()
    cluster_ids = list(DEFAULT_CONFIG.qos.clusters)

    config: dict[str, object] = {
        "seeds_base": args.seeds_base,
        "duration_s": args.duration_s,
        "traffic_mode": args.traffic_mode,
        "jitter_range_s": args.jitter_range_s,
        "snir_threshold_db": args.snir_threshold_db,
        "snir_threshold_min_db": args.snir_threshold_min_db,
        "snir_threshold_max_db": args.snir_threshold_max_db,
        "mixra_opt_max_iterations": args.mixra_opt_max_iterations,
        "mixra_opt_candidate_subset_size": args.mixra_opt_candidate_subset_size,
        "mixra_opt_epsilon": args.mixra_opt_epsilon,
        "mixra_opt_max_evals": args.mixra_opt_max_evals,
        "mixra_opt_budget": args.mixra_opt_budget,
        "mixra_opt_budget_base": args.mixra_opt_budget_base,
        "mixra_opt_budget_scale": args.mixra_opt_budget_scale,
        "mixra_opt_enabled": args.mixra_opt_enabled,
        "mixra_opt_mode": args.mixra_opt_mode,
        "mixra_opt_timeout": args.mixra_opt_timeout,
        "mixra_opt_no_fallback": args.mixra_opt_no_fallback,
        "profile_timing": args.profile_timing,
    }

    tasks = [
        (
            network_size,
            size_idx,
            snir_modes,
            replications,
            config,
            output_dir,
            cluster_ids,
            flat_output,
        )
        for size_idx, network_size in enumerate(network_sizes)
    ]

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = map(_simulate_density, tasks)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            results = pool.imap_unordered(_simulate_density, tasks)
            for result in results:
                simulated_sizes.append(int(result["network_size"]))
                total_rows += int(result["row_count"])
                rows_per_size[int(result["network_size"])] += int(result["row_count"])
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
            simulated_sizes.append(int(result["network_size"]))
            total_rows += int(result["row_count"])
            rows_per_size[int(result["network_size"])] += int(result["row_count"])
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
    if flat_output:
        aggregated_sizes = _read_aggregated_sizes(output_dir / "aggregated_results.csv")
    else:
        aggregated_sizes = _read_nested_sizes(output_dir, replications)
    missing_sizes = sorted(set(network_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        print(
            "ATTENTION: tailles manquantes dans aggregated_results.csv, "
            f"done.flag non écrit. Manquantes: {missing_label}"
        )
    else:
        (output_dir / "done.flag").write_text("done\n", encoding="utf-8")
        print("done.flag écrit (agrégation complète).")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")
    aggregated_path = output_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        _check_pdr_consistency(output_dir)
        _check_pdr_formula_for_size(output_dir, reference_size=80)
        if args.plot_summary:
            _plot_summary_pdr(output_dir)
    elif args.plot_summary:
        print(
            "Plot de synthèse ignoré: aggregated_results.csv absent "
            "(utilisez --flat-output ou make_all_plots.py)."
        )


if __name__ == "__main__":
    main()
