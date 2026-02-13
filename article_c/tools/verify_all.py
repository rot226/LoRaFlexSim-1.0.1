"""Vérifications globales des résultats/figures de l'article C.

Échec (code non-zéro) si une condition bloquante est détectée via exception explicite :
- tailles réseau attendues manquantes (Step1/Step2),
- réplications attendues manquantes (rep_0..rep_{R-1}) pour chaque taille,
- CSV vides ou sans ligne de données,
- PNG vides/corrompus,
- figures requises absentes, sans légende ou largeur > 15 pouces,
- traces de crash détectées dans les logs pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
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
CRASH_SIGNATURES: tuple[str, ...] = ("Traceback", "RuntimeError", "TypeError")
LEGEND_CHECK_REPORT_PATH = BASE_DIR / "legend_check_report.csv"
SIMULATION_QUALITY_REPORT_PATH = BASE_DIR / "simulation_quality_verify_all.json"
ALGO_LEGEND_KEYWORDS = ("adr", "mixra", "ucb1")
STEP2_REQUIRED_LEGEND_MODULES = (
    "article_c.step2.plots.plot_RL1",
    "article_c.step2.plots.plot_RL2",
    "article_c.step2.plots.plot_RL3",
    "article_c.step2.plots.plot_RL4",
    "article_c.step2.plots.plot_RL5",
    "article_c.step2.plots.plot_RL6_cluster_outage_vs_density",
)


class VerificationError(RuntimeError):
    """Erreur explicite de vérification pipeline."""


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
        "--replications",
        type=int,
        default=5,
        help="Nombre attendu de réplications par taille (rep_0..rep_{R-1}).",
    )
    parser.add_argument(
        "--skip-render-check",
        action="store_true",
        help="N'exécute pas les modules de plots pour les contrôles légende/dimension.",
    )
    parser.add_argument(
        "--success-rate-mean-min-n80",
        type=float,
        default=0.80,
        help="Seuil minimum de success_rate_mean pour la taille 80.",
    )
    parser.add_argument(
        "--collision-dominance-max",
        type=float,
        default=0.70,
        help="Seuil maximum de dominance des collisions (collisions / pertes totales).",
    )
    parser.add_argument(
        "--ucb1-non-zero-success-ratio-min",
        type=float,
        default=0.80,
        help="Seuil minimum de ratio des lignes UCB1 avec success_rate_mean > 0.",
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


def _check_replication_dirs(expected_replications: int) -> None:
    if expected_replications <= 0:
        raise VerificationError(
            f"Valeur invalide pour --replications={expected_replications} (attendu > 0)."
        )
    expected_reps = {f"rep_{idx}" for idx in range(expected_replications)}
    for step in ("step1", "step2"):
        results_dir = BASE_DIR / step / "results"
        size_dirs = _iter_nested_size_dirs(results_dir)
        for size in EXPECTED_SIZES:
            size_dir = size_dirs.get(size)
            if size_dir is None:
                raise VerificationError(
                    f"{step}: dossier de taille manquant: {results_dir / 'by_size' / f'size_{size}'} "
                    f"(ou {results_dir / f'size_{size}'})."
                )
            found = {
                path.name
                for path in size_dir.glob("rep_*")
                if path.is_dir() and re.fullmatch(r"rep_\d+", path.name)
            }
            missing = sorted(expected_reps - found)
            if missing:
                raise VerificationError(
                    f"{step}: réplications manquantes pour {size_dir}: {missing} "
                    f"(attendues: rep_0..rep_{expected_replications - 1})."
                )


def _check_step_sizes_completeness() -> list[str]:
    failures: list[str] = []
    expected = set(EXPECTED_SIZES)
    for step in ("step1", "step2"):
        aggregated = BASE_DIR / step / "results" / "aggregates" / "aggregated_results.csv"
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
        png_path = output_dir / f"{stem}.png"
        if not png_path.exists():
            candidates = [output_dir / f"{stem}.{fmt}" for fmt in formats]
            all_candidates = [png_path, *candidates]
            unique_candidates = list(dict.fromkeys(all_candidates))
            rel_candidates = ", ".join(
                str(path.relative_to(BASE_DIR)) for path in unique_candidates
            )
            failures.append(
                "Figure attendue absente (PNG requis) pour "
                f"{module_path}: {rel_candidates}"
            )
    return failures


def _assert_non_empty_csv(path: Path) -> None:
    if not path.exists():
        raise VerificationError(f"CSV introuvable: {path}")
    if path.stat().st_size <= 0:
        raise VerificationError(f"CSV vide (taille fichier nulle): {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            first_row = next(reader)
        except StopIteration as exc:
            raise VerificationError(f"CSV sans ligne de données: {path}") from exc
        if not first_row:
            raise VerificationError(f"CSV sans colonnes exploitables: {path}")


def _check_non_empty_csv_files() -> None:
    csv_paths = sorted(BASE_DIR.glob("**/*.csv"))
    for csv_path in csv_paths:
        _assert_non_empty_csv(csv_path)


def _check_png_files_valid() -> None:
    for png_path in sorted(BASE_DIR.glob("**/*.png")):
        if png_path.stat().st_size <= 0:
            raise VerificationError(f"PNG vide (taille fichier nulle): {png_path}")
        try:
            _ = plt.imread(png_path)
        except Exception as exc:
            raise VerificationError(f"PNG corrompu/non lisible: {png_path} ({exc})") from exc


def _iter_pipeline_log_paths() -> list[Path]:
    patterns = (
        "pipeline*.log",
        "*pipeline*.txt",
        "logs/**/*.log",
        "logs/**/*.txt",
        "results/**/*.log",
        "results/**/*.txt",
    )
    all_paths: set[Path] = set()
    for pattern in patterns:
        all_paths.update(path for path in BASE_DIR.glob(pattern) if path.is_file())
    return sorted(all_paths)


def _check_pipeline_logs_for_crash_traces() -> None:
    for log_path in _iter_pipeline_log_paths():
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        for signature in CRASH_SIGNATURES:
            if signature in content:
                raise VerificationError(
                    f"Trace de crash détectée dans {log_path}: signature '{signature}'."
                )


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


def _is_algorithm_legend_label(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", label.strip().lower())
    return bool(normalized) and any(
        keyword in normalized for keyword in ALGO_LEGEND_KEYWORDS
    )


def _count_algorithm_legend_entries(fig: plt.Figure) -> int:
    count = 0
    for ax in fig.axes:
        _, labels = ax.get_legend_handles_labels()
        count += sum(
            1 for label in labels if label and _is_algorithm_legend_label(label)
        )
    for legend in fig.legends:
        count += sum(
            1
            for text in legend.get_texts()
            if text.get_text() and _is_algorithm_legend_label(text.get_text())
        )
    return count


def _write_legend_check_report(rows: list[dict[str, object]]) -> None:
    with LEGEND_CHECK_REPORT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["module", "status", "legend_entries"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["module"])))


def _check_legends_and_sizes() -> list[str]:
    failures: list[str] = []
    modules = [*PLOT_MODULES["step1"], *PLOT_MODULES["step2"], *POST_PLOT_MODULES]
    legend_report_rows: list[dict[str, object]] = []
    legend_validity_by_module: dict[str, bool] = {}

    original_close = plt.close

    def _noop_close(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)

    plt.close = _noop_close
    try:
        for module_path in modules:
            before = set(plt.get_fignums())
            module = importlib.import_module(module_path)
            module_legend_entries = 0
            module_has_valid_legend = True
            try:
                _invoke_module_main(module)
            except Exception as exc:
                failures.append(f"{module_path}: exécution impossible ({exc})")
                legend_validity_by_module[module_path] = False
                legend_report_rows.append(
                    {
                        "module": module_path,
                        "status": "FAIL",
                        "legend_entries": 0,
                    }
                )
                continue

            new_numbers = [num for num in plt.get_fignums() if num not in before]
            if not new_numbers:
                failures.append(f"{module_path}: aucune figure détectée pendant l'exécution.")
                legend_validity_by_module[module_path] = False
                legend_report_rows.append(
                    {
                        "module": module_path,
                        "status": "FAIL",
                        "legend_entries": 0,
                    }
                )
                continue

            for idx, fig_no in enumerate(new_numbers, start=1):
                fig = plt.figure(fig_no)
                context = f"{module_path}#fig{idx}"
                has_legend = bool(fig.legends) or any(
                    ax.get_legend() is not None for ax in fig.axes
                )
                legend_entries = _count_algorithm_legend_entries(fig)
                module_legend_entries += legend_entries
                if not has_legend or legend_entries == 0:
                    reason = (
                        "légende absente" if not has_legend else "aucune entrée d'algorithme"
                    )
                    failures.append(f"{context}: {reason}.")
                    module_has_valid_legend = False

                width_in, height_in = fig.get_size_inches()
                if width_in <= 0 or height_in <= 0:
                    failures.append(
                        f"{context}: dimension invalide ({width_in:.2f}x{height_in:.2f} in)."
                    )
                if width_in > 15.0:
                    failures.append(
                        f"{context}: largeur de figure > 15 in ({width_in:.2f} in)."
                    )
            legend_validity_by_module[module_path] = module_has_valid_legend
            legend_report_rows.append(
                {
                    "module": module_path,
                    "status": "PASS" if module_has_valid_legend else "FAIL",
                    "legend_entries": module_legend_entries,
                }
            )
    finally:
        plt.close = original_close
        plt.close("all")

    _write_legend_check_report(legend_report_rows)

    missing_required_step2_legends = [
        module
        for module in STEP2_REQUIRED_LEGEND_MODULES
        if module in legend_validity_by_module and not legend_validity_by_module[module]
    ]
    if missing_required_step2_legends:
        failures.append(
            "Step2: sortie finale bloquée, légende invalide pour: "
            + ", ".join(missing_required_step2_legends)
        )

    return failures


def _to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assess_simulation_quality(args: argparse.Namespace) -> dict[str, object]:
    thresholds = {
        "success_rate_mean_min_n80": float(args.success_rate_mean_min_n80),
        "collision_dominance_max": float(args.collision_dominance_max),
        "ucb1_non_zero_success_ratio_min": float(args.ucb1_non_zero_success_ratio_min),
    }
    metrics: dict[str, float | None] = {
        "success_rate_mean_n80": None,
        "collision_dominance": None,
        "ucb1_non_zero_success_ratio": None,
    }
    reasons: list[str] = []
    actions: list[str] = []

    diagnostics_path = BASE_DIR / "step2" / "results" / "aggregates" / "diagnostics_step2_by_size.csv"
    diagnostics_rows = _read_csv_rows(diagnostics_path)
    for row in diagnostics_rows:
        size = _to_int(row.get("network_size") or row.get("density"))
        if size != 80:
            continue
        metrics["success_rate_mean_n80"] = _to_float(row.get("success_rate_mean"))
        break

    success_n80 = metrics["success_rate_mean_n80"]
    if success_n80 is None:
        reasons.append("Mesure success_rate_mean_n80 indisponible.")
        actions.append(
            "Vérifier la présence de diagnostics_step2_by_size.csv et l'entrée network_size=80."
        )
    elif success_n80 < thresholds["success_rate_mean_min_n80"]:
        reasons.append(
            "success_rate_mean_n80 "
            f"{success_n80:.4f} < seuil {thresholds['success_rate_mean_min_n80']:.4f}."
        )
        actions.append(
            "Relancer Step2 pour la taille 80 (plus de réplications / seeds) et inspecter les causes de pertes."
        )

    losses_path = BASE_DIR / "step2" / "results" / "aggregates" / "loss_causes_histogram.csv"
    losses_rows = _read_csv_rows(losses_path)
    if losses_rows:
        by_cause: dict[str, int] = {}
        for row in losses_rows:
            cause = str(row.get("cause", "")).strip().lower()
            count = _to_int(row.get("count")) or 0
            if cause:
                by_cause[cause] = by_cause.get(cause, 0) + count
        total_losses = sum(by_cause.values())
        collision_losses = by_cause.get("collisions", 0)
        metrics["collision_dominance"] = (
            collision_losses / total_losses if total_losses > 0 else 0.0
        )
    collision_dominance = metrics["collision_dominance"]
    if collision_dominance is None:
        reasons.append("Mesure collision_dominance indisponible.")
        actions.append(
            "Générer/valider loss_causes_histogram.csv pour confirmer la répartition des pertes."
        )
    elif collision_dominance > thresholds["collision_dominance_max"]:
        reasons.append(
            "collision_dominance "
            f"{collision_dominance:.4f} > seuil {thresholds['collision_dominance_max']:.4f}."
        )
        actions.append(
            "Réduire la contention radio (duty-cycle/charge), ajuster ADR ou stratégie d'accès pour diminuer les collisions."
        )

    aggregated_path = BASE_DIR / "step2" / "results" / "aggregates" / "aggregated_results.csv"
    aggregated_rows = _read_csv_rows(aggregated_path)
    ucb1_rows = [
        row for row in aggregated_rows if str(row.get("algorithm", "")).strip().lower() == "ucb1"
    ]
    if ucb1_rows:
        non_zero = 0
        for row in ucb1_rows:
            success_value = _to_float(row.get("success_rate_mean"))
            if success_value is not None and success_value > 0.0:
                non_zero += 1
        metrics["ucb1_non_zero_success_ratio"] = non_zero / len(ucb1_rows)
    ucb1_ratio = metrics["ucb1_non_zero_success_ratio"]
    if ucb1_ratio is None:
        reasons.append("Mesure ucb1_non_zero_success_ratio indisponible.")
        actions.append(
            "Vérifier aggregated_results.csv et la colonne algorithm pour les lignes UCB1."
        )
    elif ucb1_ratio < thresholds["ucb1_non_zero_success_ratio_min"]:
        reasons.append(
            "ucb1_non_zero_success_ratio "
            f"{ucb1_ratio:.4f} < seuil {thresholds['ucb1_non_zero_success_ratio_min']:.4f}."
        )
        actions.append(
            "Réviser la configuration UCB1 (exploration/récompense) et augmenter la durée de convergence."
        )

    quality = "low" if reasons else "ok"
    summary: dict[str, object] = {
        "simulation_quality": quality,
        "thresholds": thresholds,
        "metrics": metrics,
        "messages": reasons,
        "actions": actions,
    }
    SIMULATION_QUALITY_REPORT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _build_parser().parse_args()
    formats = _parse_formats(args.formats)

    try:
        separate_size_failures = _check_separate_size_dirs()
        if separate_size_failures:
            raise VerificationError("\n".join(separate_size_failures))
        size_completeness_failures = _check_step_sizes_completeness()
        if size_completeness_failures:
            raise VerificationError("\n".join(size_completeness_failures))
        _check_replication_dirs(args.replications)
        _check_non_empty_csv_files()
        _check_png_files_valid()

        expected_files_failures = _check_expected_files(formats)
        if expected_files_failures:
            raise VerificationError("\n".join(expected_files_failures))

        if not args.skip_render_check:
            render_failures = _check_legends_and_sizes()
            if render_failures:
                raise VerificationError("\n".join(render_failures))

        _check_pipeline_logs_for_crash_traces()
    except VerificationError as exc:
        print("FAIL")
        print(f"- {exc}")
        return 1

    quality_summary = _assess_simulation_quality(args)
    quality = str(quality_summary.get("simulation_quality", "ok"))

    print("PASS")
    print(f"simulation_quality={quality}")
    if quality == "low":
        print("QUALITÉ SIMULATION: LOW (figures générées mais interprétation prudente)")
        for message in quality_summary.get("messages", []):
            print(f"- {message}")
        for action in quality_summary.get("actions", []):
            print(f"  action: {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
