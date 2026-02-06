"""Contrôle rapide des résultats agrégés et des figures pour l'article C."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class MetricCheck:
    label: str
    keys: tuple[str, ...]
    variance: float | None = None
    values: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def compute(self, variance_floor: float) -> None:
        if len(self.values) < 2:
            self.errors.append("valeurs insuffisantes pour calculer la variance.")
            return
        self.variance = statistics.pvariance(self.values)
        if self.variance <= variance_floor:
            self.errors.append(
                f"variance trop faible (<= {variance_floor:g})."
            )


@dataclass
class FigureCheck:
    output_dir: Path
    png_files: dict[str, Path] = field(default_factory=dict)
    pdf_files: dict[str, Path] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass
class StepSummary:
    label: str
    csv_path: Path
    plots_dir: Path
    metrics: list[MetricCheck]
    figure_check: FigureCheck | None = None
    csv_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.csv_errors:
            return False
        if any(metric.errors for metric in self.metrics):
            return False
        if self.figure_check and self.figure_check.issues:
            return False
        return True


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _extract_metric_values(
    rows: Iterable[dict[str, str]],
    keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in keys:
            if key not in row:
                continue
            parsed = _parse_float(row.get(key))
            if parsed is None:
                continue
            values.append(parsed)
            break
    return values


def _collect_figures(output_dir: Path) -> FigureCheck:
    check = FigureCheck(output_dir=output_dir)
    if not output_dir.exists():
        check.issues.append(f"répertoire manquant: {output_dir}.")
        return check
    png_paths = sorted(output_dir.rglob("*.png"))
    pdf_paths = sorted(output_dir.rglob("*.pdf"))
    check.png_files = {path.stem: path for path in png_paths}
    check.pdf_files = {path.stem: path for path in pdf_paths}
    if not png_paths:
        check.issues.append("aucun PNG trouvé.")
    if not pdf_paths:
        check.issues.append("aucun PDF trouvé.")
    return check


def _read_png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except OSError:
        return None


_MEDIABOX_PATTERN = re.compile(
    rb"/MediaBox\s*\[\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
    rb"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\]"
)


def _read_pdf_dimensions(path: Path) -> tuple[float, float] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    match = _MEDIABOX_PATTERN.search(data)
    if not match:
        return None
    x0, y0, x1, y1 = (float(group) for group in match.groups())
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return None
    return width, height


def _check_figure_dimensions(check: FigureCheck, aspect_tolerance: float) -> None:
    if check.issues:
        return
    if not check.png_files or not check.pdf_files:
        return
    for stem, png_path in check.png_files.items():
        pdf_path = check.pdf_files.get(stem)
        if pdf_path is None:
            check.issues.append(f"PDF manquant pour {png_path.name}.")
            continue
        png_dims = _read_png_dimensions(png_path)
        pdf_dims = _read_pdf_dimensions(pdf_path)
        if png_dims is None:
            check.issues.append(f"dimensions PNG illisibles: {png_path.name}.")
            continue
        if pdf_dims is None:
            check.issues.append(f"dimensions PDF illisibles: {pdf_path.name}.")
            continue
        png_ratio = png_dims[0] / png_dims[1] if png_dims[1] else 0.0
        pdf_ratio = pdf_dims[0] / pdf_dims[1] if pdf_dims[1] else 0.0
        if not math.isfinite(png_ratio) or not math.isfinite(pdf_ratio):
            check.issues.append(f"ratio invalide pour {stem}.")
            continue
        if abs(png_ratio - pdf_ratio) > aspect_tolerance:
            check.issues.append(
                f"ratio PNG/PDF divergent pour {stem} "
                f"(png={png_ratio:.3f}, pdf={pdf_ratio:.3f})."
            )
    for stem, pdf_path in check.pdf_files.items():
        if stem not in check.png_files:
            check.issues.append(f"PNG manquant pour {pdf_path.name}.")


def _check_step(
    label: str,
    csv_path: Path,
    plots_dir: Path,
    variance_floor: float,
    aspect_tolerance: float,
) -> StepSummary:
    metrics = [
        MetricCheck("reward", ("reward_mean", "reward")),
        MetricCheck("success_rate", ("success_rate_mean", "success_rate", "success_mean")),
    ]
    summary = StepSummary(label=label, csv_path=csv_path, plots_dir=plots_dir, metrics=metrics)
    if not csv_path.exists():
        summary.csv_errors.append(f"CSV manquant: {csv_path}.")
        summary.figure_check = _collect_figures(plots_dir)
        if summary.figure_check:
            _check_figure_dimensions(summary.figure_check, aspect_tolerance)
        return summary
    rows = _load_csv_rows(csv_path)
    if not rows:
        summary.csv_errors.append(f"CSV vide: {csv_path}.")
    for metric in metrics:
        metric.values = _extract_metric_values(rows, metric.keys)
        if not metric.values:
            metric.errors.append(
                f"aucune colonne disponible parmi: {', '.join(metric.keys)}."
            )
        else:
            metric.compute(variance_floor)
    summary.figure_check = _collect_figures(plots_dir)
    _check_figure_dimensions(summary.figure_check, aspect_tolerance)
    return summary


def _print_summary(summary: StepSummary) -> None:
    status = "PASS" if summary.passed else "FAIL"
    print(f"\n[{summary.label}] {status}")
    print(f"- CSV: {summary.csv_path}")
    if summary.csv_errors:
        for error in summary.csv_errors:
            print(f"  - ERREUR: {error}")
    for metric in summary.metrics:
        if metric.variance is not None:
            variance_label = f"{metric.variance:.6g}"
        else:
            variance_label = "N/A"
        metric_status = "OK" if not metric.errors else "FAIL"
        print(f"- Variance {metric.label}: {variance_label} ({metric_status})")
        for error in metric.errors:
            print(f"  - {error}")
    if summary.figure_check:
        fig_status = "OK" if not summary.figure_check.issues else "FAIL"
        print(f"- Figures: {summary.figure_check.output_dir} ({fig_status})")
        for issue in summary.figure_check.issues:
            print(f"  - {issue}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Contrôle rapide des CSV agrégés (variance reward/success_rate) "
            "et des exports PNG/PDF."
        )
    )
    parser.add_argument(
        "--step1-csv",
        type=Path,
        default=Path("article_c/step1/results/aggregated_results.csv"),
        help="Chemin du CSV agrégé Step1.",
    )
    parser.add_argument(
        "--step2-csv",
        type=Path,
        default=Path("article_c/step2/results/aggregated_results.csv"),
        help="Chemin du CSV agrégé Step2.",
    )
    parser.add_argument(
        "--step1-plots",
        type=Path,
        default=Path("article_c/step1/plots/output"),
        help="Répertoire des figures Step1.",
    )
    parser.add_argument(
        "--step2-plots",
        type=Path,
        default=Path("article_c/step2/plots/output"),
        help="Répertoire des figures Step2.",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-9,
        help="Seuil minimal pour considérer la variance non nulle.",
    )
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=0.02,
        help="Tolérance sur le ratio largeur/hauteur PNG vs PDF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summaries = [
        _check_step(
            "Step1",
            args.step1_csv,
            args.step1_plots,
            args.variance_floor,
            args.aspect_tolerance,
        ),
        _check_step(
            "Step2",
            args.step2_csv,
            args.step2_plots,
            args.variance_floor,
            args.aspect_tolerance,
        ),
    ]
    for summary in summaries:
        _print_summary(summary)
    return 0 if all(summary.passed for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
