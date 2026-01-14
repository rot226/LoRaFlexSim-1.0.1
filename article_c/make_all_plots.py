"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path


PLOT_MODULES = {
    "step1": [
        "article_c.step1.plots.plot_S1",
        "article_c.step1.plots.plot_S2",
        "article_c.step1.plots.plot_S3",
        "article_c.step1.plots.plot_S4",
        "article_c.step1.plots.plot_S5",
        "article_c.step1.plots.plot_S6",
        "article_c.step1.plots.plot_S6_cluster_pdr_vs_density",
        "article_c.step1.plots.plot_S7_cluster_outage_vs_density",
    ],
    "step2": [
        "article_c.step2.plots.plot_RL1",
        "article_c.step2.plots.plot_RL2",
        "article_c.step2.plots.plot_RL3",
        "article_c.step2.plots.plot_RL4",
        "article_c.step2.plots.plot_RL5",
    ],
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour générer les figures."""
    parser = argparse.ArgumentParser(
        description="Génère toutes les figures à partir des CSV agrégés."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="step1,step2",
        help="Étapes à tracer (ex: step1,step2).",
    )
    return parser


def _parse_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [step for step in steps if step not in PLOT_MODULES]
    if unknown:
        raise ValueError(f"Étape(s) inconnue(s): {', '.join(unknown)}")
    return steps


def _run_plot_module(module_path: str) -> None:
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_path} sans fonction main().")
    module.main()


def _validate_snir_mode_column(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
        if "snir_mode" not in fieldnames:
            raise ValueError(
                f"Le CSV {path} doit contenir une colonne 'snir_mode'."
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    article_dir = Path(__file__).resolve().parent
    csv_paths: list[Path] = []
    if "step1" in steps:
        csv_paths.append(article_dir / "step1" / "results" / "aggregated_results.csv")
        csv_paths.append(article_dir / "step2" / "results" / "aggregated_results.csv")
    if "step2" in steps and "step1" not in steps:
        csv_paths.append(article_dir / "step2" / "results" / "aggregated_results.csv")
    _validate_snir_mode_column(csv_paths)
    for step in steps:
        for module_path in PLOT_MODULES[step]:
            _run_plot_module(module_path)


if __name__ == "__main__":
    main()
