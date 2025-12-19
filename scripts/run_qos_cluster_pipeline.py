"""Pipeline complète pour exécuter le banc QoS et générer les graphes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.scenarios.qos_cluster_bench import (  # noqa: E402
    ALGORITHMS,
    DEFAULT_RESULTS_DIR,
    run_bench,
)
from loraflexsim.scenarios.qos_cluster_presets import (  # noqa: E402
    describe_presets,
    get_preset,
    list_presets,
)
from scripts import qos_cluster_plots  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=[preset.name for preset in list_presets()],
        default="quick",
        help="Préréglage de scénarios à exécuter (défaut : quick)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Graine de simulation initiale",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Solveur à utiliser pour MixRA-Opt",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée maximale de simulation en secondes (par défaut : valeur du preset)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Répertoire racine pour les résultats CSV",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Répertoire racine pour les figures",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="N'exécute pas la génération des figures",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les impressions de progression",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Affiche les préréglages disponibles et quitte",
    )
    return parser


def _resolve_directories(
    preset_name: str,
    results_dir: Path | None,
    figures_dir: Path | None,
) -> tuple[Path, Path]:
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR / preset_name
    if figures_dir is None:
        figures_dir = qos_cluster_plots.DEFAULT_FIGURES_DIR / preset_name
    return results_dir, figures_dir


def _print_summary(summary: Mapping[str, Any]) -> None:
    states = summary.get("states")
    if isinstance(states, Mapping):
        for label, payload in states.items():
            report_path = payload.get("report_path") if isinstance(payload, Mapping) else None
            summary_path = payload.get("summary_path") if isinstance(payload, Mapping) else None
            print(f"État SNIR : {label}")
            if report_path:
                print(f"  Rapport Markdown : {report_path}")
            if summary_path:
                print(f"  Résumé JSON : {summary_path}")
        return

    report_path = summary.get("report_path")
    summary_path = summary.get("summary_path")
    if report_path:
        print(f"Rapport Markdown : {report_path}")
    if summary_path:
        print(f"Résumé JSON : {summary_path}")


def main(
    argv: list[str] | None = None,
    *,
    runner=run_bench,
    plotter=qos_cluster_plots.generate_plots,
) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        print(describe_presets(len(ALGORITHMS)))
        return {}

    preset = get_preset(args.preset)
    results_dir, figures_dir = _resolve_directories(args.preset, args.results_dir, args.figures_dir)
    duration = args.duration if args.duration is not None else preset.simulation_duration_s

    if not args.quiet:
        print(f"Preset sélectionné : {preset.label}")
        print(
            "Charges : "
            + ", ".join(str(value) for value in preset.node_counts)
            + " | Périodes : "
            + ", ".join(
                f"{int(value) if float(value).is_integer() else value:g} s" for value in preset.tx_periods
            )
        )
        print(f"Durée max : {duration / 3600:.1f} h")
        print(f"Résultats : {results_dir}")
        if not args.skip_plots:
            print(f"Figures : {figures_dir}")

    summary = runner(
        node_counts=preset.node_counts,
        tx_periods=preset.tx_periods,
        seed=args.seed,
        output_dir=results_dir,
        simulation_duration_s=duration,
        mixra_solver=args.mixra_solver,
        quiet=args.quiet,
        progress_callback=None if args.quiet else None,
    )

    if not args.quiet:
        _print_summary(summary)

    if not args.skip_plots:
        generated_any = False
        state_payloads = summary.get("states") if isinstance(summary, Mapping) else None
        if isinstance(state_payloads, Mapping) and state_payloads:
            for label in state_payloads:
                state_dir = results_dir / label
                state_fig_dir = figures_dir / label
                generated = plotter(state_dir, state_fig_dir)
                generated_any = generated_any or bool(generated)
                if not args.quiet:
                    suffix = f" ({label})"
                    if generated:
                        print(f"Figures enregistrées dans {state_fig_dir}{suffix}")
                    else:
                        print(f"Aucune figure générée pour {state_dir}{suffix}.")
        else:
            generated_any = plotter(results_dir, figures_dir)
            if not args.quiet:
                if generated_any:
                    print(f"Figures enregistrées dans {figures_dir}")
                else:
                    print("Aucun résultat disponible pour générer des figures.")
        if not args.quiet and not generated_any:
            print("Aucune figure n'a été produite.")
    return summary


if __name__ == "__main__":
    main()
