"""Vérifications globales des résultats/figures de l'article C.

Échec (code non-zéro) si l'une des conditions suivantes est détectée :
- Step1/aggregated_results.csv contient moins de 2 tailles réseau distinctes.
- Une figure attendue est absente.
- Une figure générée ne contient pas de légende.
- Dimension anormale (>12 in) pour une figure mono-panel.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
from importlib.util import find_spec

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from article_c.common.config import BASE_DIR
from article_c.make_all_plots import (
    MANIFEST_STEP_OUTPUT_DIRS,
    PLOT_MODULES,
    POST_PLOT_MODULES,
    _extract_plot_metadata,
)

SUPPORTED_FORMATS = ("png", "pdf", "eps", "svg")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vérifie les CSV/figures (présence, légende, dimensions)."
    )
    parser.add_argument(
        "--formats",
        default=",".join(SUPPORTED_FORMATS),
        help="Formats acceptés pour valider la présence des figures (csv).",
    )
    parser.add_argument(
        "--skip-render-check",
        action="store_true",
        help="N'exécute pas les modules de plots pour les contrôles légende/dimension.",
    )
    return parser


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(
        fmt.strip().lower()
        for fmt in str(raw).split(",")
        if fmt.strip()
    )
    return formats or SUPPORTED_FORMATS


def _read_step1_sizes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    sizes: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_size = row.get("network_size", "")
            try:
                size = int(float(str(raw_size).strip()))
            except ValueError:
                continue
            if size > 0:
                sizes.add(size)
    return sizes


def _iter_expected_figures() -> list[tuple[str, Path, str]]:
    expected: list[tuple[str, Path, str]] = []
    modules_by_step = {
        **{module_path: "step1" for module_path in PLOT_MODULES["step1"]},
        **{module_path: "step2" for module_path in PLOT_MODULES["step2"]},
        **{module_path: "post" for module_path in POST_PLOT_MODULES},
    }
    for module_path, step in modules_by_step.items():
        _, _, _, stems = _extract_plot_metadata(module_path)
        output_dir = MANIFEST_STEP_OUTPUT_DIRS[step]
        for stem in stems:
            expected.append((module_path, output_dir, stem))
    return expected


def _check_expected_files(formats: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for module_path, output_dir, stem in _iter_expected_figures():
        candidates = [output_dir / f"{stem}.{fmt}" for fmt in formats]
        if not any(path.exists() for path in candidates):
            rel_candidates = ", ".join(str(path.relative_to(BASE_DIR)) for path in candidates)
            failures.append(
                f"Figure attendue absente pour {module_path}: {rel_candidates}"
            )
    return failures


def _invoke_module_main(module: ModuleType) -> None:
    if not hasattr(module, "main"):
        raise AttributeError(f"{module.__name__} ne définit pas main().")
    signature = inspect.signature(module.main)
    kwargs: dict[str, object] = {}
    if "allow_sample" in signature.parameters:
        kwargs["allow_sample"] = True
    if "enable_suptitle" in signature.parameters:
        kwargs["enable_suptitle"] = False
    module.main(**kwargs) if kwargs else module.main()


def _check_legends_and_sizes() -> list[str]:
    failures: list[str] = []
    modules = [*PLOT_MODULES["step1"], *PLOT_MODULES["step2"], *POST_PLOT_MODULES]

    original_close = plt.close

    def _noop_close(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)

    plt.close = _noop_close
    try:
        for module_path in modules:
            before = set(plt.get_fignums())
            module = importlib.import_module(module_path)
            try:
                _invoke_module_main(module)
            except Exception as exc:
                failures.append(f"{module_path}: exécution impossible ({exc})")
                continue

            new_numbers = [num for num in plt.get_fignums() if num not in before]
            if not new_numbers:
                failures.append(f"{module_path}: aucune figure détectée pendant l'exécution.")
                continue

            for idx, fig_no in enumerate(new_numbers, start=1):
                fig = plt.figure(fig_no)
                context = f"{module_path}#fig{idx}"
                has_legend = bool(fig.legends) or any(
                    ax.get_legend() is not None for ax in fig.axes
                )
                if not has_legend:
                    failures.append(f"{context}: légende absente.")

                width_in, height_in = fig.get_size_inches()
                # Règle demandée: single-plot anormal si >12 in.
                if len(fig.axes) <= 1 and (width_in > 12.0 or height_in > 12.0):
                    failures.append(
                        f"{context}: dimension anormale pour mono-panel "
                        f"({width_in:.2f}x{height_in:.2f} in)."
                    )
    finally:
        plt.close = original_close
        plt.close("all")

    return failures


def main() -> int:
    args = _build_parser().parse_args()
    formats = _parse_formats(args.formats)

    failures: list[str] = []

    step1_agg = BASE_DIR / "step1" / "results" / "aggregated_results.csv"
    sizes = _read_step1_sizes(step1_agg)
    if len(sizes) < 2:
        failures.append(
            "Step1 aggregated_results.csv invalide: moins de 2 tailles réseau distinctes."
        )

    failures.extend(_check_expected_files(formats))

    if not args.skip_render_check:
        failures.extend(_check_legends_and_sizes())

    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
