"""Exécute toutes les étapes de l'article C."""

from __future__ import annotations

import argparse

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.utils import parse_network_size_list
from article_c.step1.run_step1 import main as run_step1
from article_c.step2.run_step2 import main as run_step2
from article_c.validate_results import main as validate_results


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour l'exécution complète."""
    parser = argparse.ArgumentParser(
        description="Exécute les étapes 1 et 2 avec des arguments communs."
    )
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=None,
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
        default=None,
        help="Nombre de réplications par configuration.",
    )
    parser.add_argument(
        "--seeds_base",
        type=int,
        default=None,
        help="Seed de base commune aux étapes 1 et 2.",
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
        default=None,
        help="Liste des modes SNIR pour l'étape 1 (ex: snir_on,snir_off).",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=None,
        help="Seuil SNIR (dB).",
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=None,
        help="Bruit thermique (dBm).",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Ajoute un timestamp dans les sorties de l'étape 2.",
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default=None,
        choices=("periodic", "poisson"),
        help="Modèle de trafic pour les étapes 1 et 2 (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range-s",
        dest="jitter_range_s",
        type=float,
        default=30.0,
        help="Amplitude du jitter pour l'étape 2 (secondes).",
    )
    parser.add_argument(
        "--jitter-range",
        dest="jitter_range_s",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias de --jitter-range-s (déprécié).",
    )
    parser.add_argument(
        "--window-duration-s",
        type=float,
        default=None,
        help="Durée d'une fenêtre de simulation (secondes).",
    )
    parser.add_argument(
        "--traffic-coeff-min",
        type=float,
        default=None,
        help="Coefficient de trafic minimal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-max",
        type=float,
        default=None,
        help="Coefficient de trafic maximal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active/désactive la variabilité de trafic par nœud.",
    )
    parser.add_argument(
        "--window-delay-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active/désactive le délai aléatoire entre fenêtres.",
    )
    parser.add_argument(
        "--window-delay-range-s",
        type=float,
        default=None,
        help="Amplitude du délai aléatoire entre fenêtres (secondes).",
    )
    parser.add_argument(
        "--step1-outdir",
        type=str,
        default=None,
        help="Répertoire de sortie de l'étape 1.",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Ignore l'exécution de l'étape 1.",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Ignore l'exécution de l'étape 2.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche la progression de l'étape 1.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Durée de la simulation pour l'étape 1 (secondes).",
    )
    parser.add_argument(
        "--mixra-opt-max-iterations",
        type=int,
        default=None,
        help="Nombre maximal d'itérations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-candidate-subset-size",
        type=int,
        default=None,
        help="Nombre maximal de nœuds optimisés par itération en MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-epsilon",
        type=float,
        default=None,
        help="Seuil d'amélioration pour la convergence MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-max-evals",
        type=int,
        default=None,
        help="Nombre maximal d'évaluations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-budget",
        type=int,
        default=None,
        help="Budget cible d'évaluations pour MixRA-Opt (max d'évaluations).",
    )
    parser.add_argument(
        "--mixra-opt-budget-base",
        type=int,
        default=None,
        help="Offset additif appliqué au budget MixRA-Opt calculé.",
    )
    parser.add_argument(
        "--mixra-opt-budget-scale",
        type=float,
        default=None,
        help="Facteur multiplicatif appliqué au budget MixRA-Opt calculé.",
    )
    parser.add_argument(
        "--mixra-opt-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active ou désactive MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-mode",
        choices=("fast", "balanced", "full"),
        default=None,
        help="Mode MixRA-Opt (balanced par défaut).",
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
        default=None,
        help="Timeout (secondes) pour MixRA-Opt afin d'éviter les blocages.",
    )
    parser.add_argument(
        "--plot-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Génère un plot de synthèse avec barres d'erreur à l'étape 1.",
    )
    parser.add_argument(
        "--profile-timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Affiche les durées des étapes internes pour l'étape 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de processus worker pour paralléliser l'étape 1.",
    )
    return parser


def _build_step1_args(args: argparse.Namespace) -> list[str]:
    step1_args: list[str] = []
    if args.network_sizes:
        step1_args.append("--network-sizes")
        step1_args.extend([str(size) for size in args.network_sizes])
    if args.replications is not None:
        step1_args.extend(["--replications", str(args.replications)])
    if args.seeds_base is not None:
        step1_args.extend(["--seeds_base", str(args.seeds_base)])
    if args.snir_modes:
        step1_args.extend(["--snir_modes", args.snir_modes])
    if args.snir_threshold_db is not None:
        step1_args.extend(["--snir-threshold-db", str(args.snir_threshold_db)])
    if args.noise_floor_dbm is not None:
        step1_args.extend(["--noise-floor-dbm", str(args.noise_floor_dbm)])
    if args.traffic_mode is not None:
        step1_args.extend(["--traffic-mode", args.traffic_mode])
    if args.jitter_range_s is not None:
        step1_args.extend(["--jitter-range-s", str(args.jitter_range_s)])
    if args.duration_s is not None:
        step1_args.extend(["--duration-s", str(args.duration_s)])
    if args.step1_outdir:
        step1_args.extend(["--outdir", args.step1_outdir])
    else:
        step1_args.extend(["--outdir", "article_c/step1/results"])
    if args.progress is not None:
        step1_args.append("--progress" if args.progress else "--no-progress")
    if args.mixra_opt_max_iterations is not None:
        step1_args.extend(
            ["--mixra-opt-max-iterations", str(args.mixra_opt_max_iterations)]
        )
    if args.mixra_opt_candidate_subset_size is not None:
        step1_args.extend(
            [
                "--mixra-opt-candidate-subset-size",
                str(args.mixra_opt_candidate_subset_size),
            ]
        )
    if args.mixra_opt_epsilon is not None:
        step1_args.extend(["--mixra-opt-epsilon", str(args.mixra_opt_epsilon)])
    if args.mixra_opt_max_evals is not None:
        step1_args.extend(["--mixra-opt-max-evals", str(args.mixra_opt_max_evals)])
    if args.mixra_opt_budget is not None:
        step1_args.extend(["--mixra-opt-budget", str(args.mixra_opt_budget)])
    if args.mixra_opt_budget_base is not None:
        step1_args.extend(["--mixra-opt-budget-base", str(args.mixra_opt_budget_base)])
    if args.mixra_opt_budget_scale is not None:
        step1_args.extend(["--mixra-opt-budget-scale", str(args.mixra_opt_budget_scale)])
    if args.mixra_opt_enabled is not None:
        step1_args.append(
            "--mixra-opt-enabled"
            if args.mixra_opt_enabled
            else "--no-mixra-opt-enabled"
        )
    if args.mixra_opt_mode is not None:
        step1_args.extend(["--mixra-opt-mode", args.mixra_opt_mode])
    if args.mixra_opt_no_fallback:
        step1_args.append("--mixra-opt-no-fallback")
    if args.mixra_opt_timeout is not None:
        step1_args.extend(["--mixra-opt-timeout", str(args.mixra_opt_timeout)])
    if args.plot_summary is not None:
        step1_args.append(
            "--plot-summary" if args.plot_summary else "--no-plot-summary"
        )
    if args.profile_timing is not None:
        step1_args.append(
            "--profile-timing" if args.profile_timing else "--no-profile-timing"
        )
    if args.workers is not None:
        step1_args.extend(["--workers", str(args.workers)])
    return step1_args


def _build_step2_args(args: argparse.Namespace) -> list[str]:
    step2_args: list[str] = []
    if args.network_sizes:
        step2_args.append("--network-sizes")
        step2_args.extend([str(size) for size in args.network_sizes])
    if args.replications is not None:
        step2_args.extend(["--replications", str(args.replications)])
    if args.seeds_base is not None:
        step2_args.extend(["--seeds_base", str(args.seeds_base)])
    if args.timestamp:
        step2_args.append("--timestamp")
    if args.snir_threshold_db is not None:
        step2_args.extend(["--snir-threshold-db", str(args.snir_threshold_db)])
    if args.noise_floor_dbm is not None:
        step2_args.extend(["--noise-floor-dbm", str(args.noise_floor_dbm)])
    if args.traffic_mode is not None:
        step2_args.extend(["--traffic-mode", args.traffic_mode])
    if args.jitter_range_s is not None:
        step2_args.extend(["--jitter-range-s", str(args.jitter_range_s)])
    if args.window_duration_s is not None:
        step2_args.extend(["--window-duration-s", str(args.window_duration_s)])
    if args.traffic_coeff_min is not None:
        step2_args.extend(["--traffic-coeff-min", str(args.traffic_coeff_min)])
    if args.traffic_coeff_max is not None:
        step2_args.extend(["--traffic-coeff-max", str(args.traffic_coeff_max)])
    if args.traffic_coeff_enabled is not None:
        step2_args.append(
            "--traffic-coeff-enabled"
            if args.traffic_coeff_enabled
            else "--no-traffic-coeff-enabled"
        )
    if args.window_delay_enabled is not None:
        step2_args.append(
            "--window-delay-enabled"
            if args.window_delay_enabled
            else "--no-window-delay-enabled"
        )
    if args.window_delay_range_s is not None:
        step2_args.extend(["--window-delay-range-s", str(args.window_delay_range_s)])
    return step2_args


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.step1_outdir is not None:
        default_step1_dir = (
            Path(__file__).resolve().parent / "step1" / "results"
        ).resolve()
        requested_dir = Path(args.step1_outdir).resolve()
        if requested_dir != default_step1_dir:
            raise ValueError(
                "Étape 1: le répertoire de sortie doit être "
                f"{default_step1_dir}."
            )
    network_sizes = (
        parse_network_size_list(args.network_sizes)
        if args.network_sizes
        else list(DEFAULT_CONFIG.scenario.network_sizes)
    )
    for size in network_sizes:
        size_args = argparse.Namespace(**vars(args))
        size_args.network_sizes = [size]
        if not size_args.skip_step1:
            run_step1(_build_step1_args(size_args))
        if not size_args.skip_step2:
            run_step2(_build_step2_args(size_args))
        print(f"Résumé: taille de réseau {size} terminée.")
    print("Validation des résultats (article C) en cours...")
    validation_code = validate_results([])
    if validation_code != 0:
        raise SystemExit(validation_code)


if __name__ == "__main__":
    main()
