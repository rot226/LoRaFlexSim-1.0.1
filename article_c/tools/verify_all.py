"""Vérifications globales des résultats/figures de l'article C.

Échec (code non-zéro) si l'une des conditions suivantes est détectée :
- Une taille attendue ne possède pas son dossier de résultats séparé.
- Step1/Step2 aggregated_results.csv ne couvrent pas toutes les tailles [80,160,320,640,1280].
- Une figure attendue est absente.
- Une figure générée ne contient pas de légende.
- Dimension anormale (>12 in) pour une figure mono-panel.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
from importlib.util import find_spec

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from article_c.common.config import BASE_DIR
from article_c.common.expected_figures import EXPECTED_FIGURES_BY_STEP
from article_c.make_all_plots import (
    MANIFEST_STEP_OUTPUT_DIRS,
    POST_PLOT_MODULES,
    PLOT_MODULES,
)

SUPPORTED_FORMATS = ("png", "pdf", "eps", "svg")
EXPECTED_SIZES: tuple[int, ...] = (80, 160, 320, 640, 1280)


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


def _read_sizes_from_aggregated(path: Path) -> set[int]:
    if not path.exists():
        return set()

    sizes: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_size = row.get("network_size")
            if raw_size in (None, ""):
                raw_size = row.get("density")
            try:
                size = int(float(str(raw_size).strip()))
            except (TypeError, ValueError):
                continue
            if size > 0:
                sizes.add(size)
    return sizes


def _iter_nested_size_dirs(results_dir: Path) -> dict[int, Path]:
    """Retourne les dossiers `size_<N>` trouvés (legacy + by_size)."""

    size_dirs: dict[int, Path] = {}
    candidates = [
        *sorted(results_dir.glob("size_*")),
        *sorted((results_dir / "by_size").glob("size_*")),
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        match = re.fullmatch(r"size_(\d+)", candidate.name)
        if not match:
            continue
        size_value = int(match.group(1))
        size_dirs[size_value] = candidate
    return size_dirs


def _check_separate_size_dirs() -> list[str]:
    failures: list[str] = []
    for step in ("step1", "step2"):
        results_dir = BASE_DIR / step / "results"
        size_dirs = _iter_nested_size_dirs(results_dir)
        for size in EXPECTED_SIZES:
            if size not in size_dirs:
                failures.append(
                    f"{step}: dossier séparé manquant pour la taille {size} "
                    f"(attendu: {results_dir / 'by_size' / f'size_{size}'} ou {results_dir / f'size_{size}'})."
                )
    return failures


def _check_step_sizes_completeness() -> list[str]:
    failures: list[str] = []
    expected = set(EXPECTED_SIZES)
    for step in ("step1", "step2"):
        aggregated = BASE_DIR / step / "results" / "aggregated_results.csv"
        found_sizes = _read_sizes_from_aggregated(aggregated)
        missing = sorted(expected - found_sizes)
        if missing:
            failures.append(
                f"{step}: tailles manquantes dans {aggregated.relative_to(BASE_DIR)}: {missing} "
                f"(attendues: {list(EXPECTED_SIZES)})."
            )
    return failures


def _iter_expected_figures() -> list[tuple[str, Path, str]]:
    expected: list[tuple[str, Path, str]] = []
    for step, module_entries in EXPECTED_FIGURES_BY_STEP.items():
        output_dir = MANIFEST_STEP_OUTPUT_DIRS[step]
        for module_path, stems in module_entries:
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
                # Règle demandée: dimensions anormales (single-plot >12 in)
                # + garde-fou dimensions non-positives.
                if width_in <= 0 or height_in <= 0:
                    failures.append(
                        f"{context}: dimension invalide ({width_in:.2f}x{height_in:.2f} in)."
                    )
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

    failures.extend(_check_separate_size_dirs())
    failures.extend(_check_step_sizes_completeness())

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
