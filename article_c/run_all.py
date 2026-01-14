"""Exécute toutes les étapes de l'article C."""

from __future__ import annotations

import argparse

from article_c.step1.run_step1 import main as run_step1
from article_c.step2.run_step2 import main as run_step2


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour l'exécution complète."""
    parser = argparse.ArgumentParser(
        description="Exécute les étapes 1 et 2 avec des arguments communs."
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=None,
        help="Liste des densités (ex: 0.1 0.5 1.0).",
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
        help="Seuil SNIR/capture (dB).",
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
        help="Modèle de trafic pour l'étape 2 (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range",
        type=float,
        default=None,
        help="Amplitude du jitter pour l'étape 2 (secondes).",
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
    return parser


def _build_step1_args(args: argparse.Namespace) -> list[str]:
    step1_args: list[str] = []
    if args.densities:
        step1_args.append("--densities")
        step1_args.extend([str(density) for density in args.densities])
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
    if args.step1_outdir:
        step1_args.extend(["--outdir", args.step1_outdir])
    return step1_args


def _build_step2_args(args: argparse.Namespace) -> list[str]:
    step2_args: list[str] = []
    if args.densities:
        step2_args.append("--densities")
        step2_args.extend([str(density) for density in args.densities])
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
    if args.jitter_range is not None:
        step2_args.extend(["--jitter-range", str(args.jitter_range)])
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
    run_step1(_build_step1_args(args))
    run_step2(_build_step2_args(args))


if __name__ == "__main__":
    main()
