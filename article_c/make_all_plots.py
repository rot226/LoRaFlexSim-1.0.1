"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import importlib


PLOT_MODULES = {
    "step1": [
        "article_c.step1.plots.plot_S1",
        "article_c.step1.plots.plot_S2",
        "article_c.step1.plots.plot_S3",
        "article_c.step1.plots.plot_S4",
        "article_c.step1.plots.plot_S5",
        "article_c.step1.plots.plot_RL1",
        "article_c.step1.plots.plot_RL2",
        "article_c.step1.plots.plot_RL3",
        "article_c.step1.plots.plot_RL4",
        "article_c.step1.plots.plot_RL5",
    ],
    "step2": [
        "article_c.step2.plots.plot_S1",
        "article_c.step2.plots.plot_S2",
        "article_c.step2.plots.plot_S3",
        "article_c.step2.plots.plot_S4",
        "article_c.step2.plots.plot_S5",
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


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    for step in steps:
        for module_path in PLOT_MODULES[step]:
            _run_plot_module(module_path)


if __name__ == "__main__":
    main()
