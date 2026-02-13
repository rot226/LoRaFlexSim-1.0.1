"""Compare les métriques SNIR on/off et superpose les courbes auteurs.

Ce script charge les CSV agrégés des étapes 1 et 2, filtre les lignes par
`snir_mode` (snir_on / snir_off), calcule DER/PDR/throughput et trace des
courbes LoRaFlexSim avec, si disponibles, des courbes auteurs.

Entrées attendues
-----------------
- Step1 agrégé : CSV `aggregated_results.csv` avec au minimum
  `network_size`, `algo`, `snir_mode` et une métrique PDR (`pdr_mean`, `pdr`, ...).
- Step2 agrégé : CSV `aggregated_results.csv` avec au minimum
  `network_size`, `algo`, `snir_mode` et une métrique de throughput
  (`throughput_success_mean`, `throughput_success`, ...).
- Courbes auteurs (optionnel) : CSV dédié avec colonnes :
  `metric` (pdr/der/throughput), `snir_mode` (snir_on/snir_off), `x`, `y`,
  `label` (optionnel), `algo` (optionnel).

Sorties produites
-----------------
Les figures sont écrites dans `--output-dir` avec les stems suivants :
- `compare_pdr_snir`  (PDR vs taille du réseau)
- `compare_der_snir`  (DER vs taille du réseau)
- `compare_throughput_snir` (throughput vs taille du réseau)

Les formats d'export sont contrôlés via `--formats` (par défaut: png).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    MetricStatus,
    SNIR_LABELS,
    SNIR_MODES,
    algo_label,
    apply_suptitle,
    apply_figure_layout,
    apply_plot_style,
    collect_legend_entries,
    deduplicate_legend_entries,
    ensure_network_size,
    fallback_legend_handles,
    filter_mixra_opt_fallback,
    is_constant_metric,
    load_step1_aggregated,
    load_step2_aggregated,
    metric_values,
    parse_export_formats,
    plot_metric_by_snir,
    render_metric_status,
    save_figure,
    set_default_export_formats,
)
from article_c.common.plotting_style import label_for
from article_c.common.plotting_style import SUPTITLE_Y

LOGGER = logging.getLogger(__name__)

METRIC_ALIASES = {
    "pdr": "pdr",
    "der": "der",
    "throughput": "throughput",
    "tp": "throughput",
}


@dataclass(frozen=True)
class AuthorCurve:
    metric: str
    snir_mode: str
    x: float
    y: float
    label: str
    algo: str | None = None


def _normalize_snir(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"snir_on", "on", "true", "1", "yes"}:
        return "snir_on"
    if text in {"snir_off", "off", "false", "0", "no"}:
        return "snir_off"
    return None


def _normalize_metric(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return METRIC_ALIASES.get(text)


def _load_author_curves(path: Path) -> list[AuthorCurve]:
    if not path.exists():
        LOGGER.info("Aucune courbe auteur trouvée: %s", path)
        return []
    curves: list[AuthorCurve] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = _normalize_metric(row.get("metric"))
            snir_mode = _normalize_snir(row.get("snir_mode"))
            if metric is None or snir_mode is None:
                continue
            try:
                x_value = float(row.get("x", ""))
                y_value = float(row.get("y", ""))
            except (TypeError, ValueError):
                continue
            label = str(row.get("label", "")).strip()
            algo_raw = str(row.get("algo", "")).strip().lower()
            algo_value = algo_raw or None
            curves.append(
                AuthorCurve(
                    metric=metric,
                    snir_mode=snir_mode,
                    x=x_value,
                    y=y_value,
                    label=label,
                    algo=algo_value,
                )
            )
    return curves


def _load_results_with_fallback(path: Path, *, step: int) -> list[dict[str, object]]:
    loader = load_step1_aggregated if step == 1 else load_step2_aggregated
    step_label = f"step{step}"

    results_dir = path.parent.parent if path.parent.name == "aggregates" else path.parent
    try:
        rows = loader(path)
        if rows:
            LOGGER.info("source utilisée (%s): %s", step_label, path)
            return rows
    except Exception as exc:  # noqa: BLE001 - fallback attendu
        LOGGER.warning("Impossible de charger %s: %s", path, exc)

    by_size_paths = sorted(results_dir.glob("by_size/size_*/aggregated_results.csv"))
    if not by_size_paths:
        return []
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer: csv.DictWriter[str] | None = None
            for by_size_path in by_size_paths:
                with by_size_path.open("r", encoding="utf-8", newline="") as src:
                    reader = csv.DictReader(src)
                    if not reader.fieldnames:
                        continue
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
    except OSError as exc:
        LOGGER.warning("Agrégation by_size impossible vers %s: %s", path, exc)
        return []

    rows = loader(path)
    if rows:
        LOGGER.info(
            "source utilisée (%s): %s/by_size/size_*/aggregated_results.csv",
            step_label,
            results_dir,
        )
    return rows


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_optional(value: float | None, target: float | None) -> bool:
    if target is None:
        return True
    if value is None:
        return False
    return math.isclose(value, target, abs_tol=1e-6)


def _filter_by_snir_threshold(
    rows: list[dict[str, object]],
    *,
    snir_threshold_db: float | None,
    snir_threshold_min_db: float | None,
    snir_threshold_max_db: float | None,
) -> list[dict[str, object]]:
    if (
        snir_threshold_db is None
        and snir_threshold_min_db is None
        and snir_threshold_max_db is None
    ):
        return rows
    has_thresholds = any(
        "snir_threshold_db" in row
        or "snir_threshold_min_db" in row
        or "snir_threshold_max_db" in row
        for row in rows
    )
    if not has_thresholds:
        LOGGER.warning(
            "Filtre SNIR ignoré (colonnes snir_threshold_* absentes dans les CSV)."
        )
        return rows
    filtered: list[dict[str, object]] = []
    for row in rows:
        if not _matches_optional(
            _float_or_none(row.get("snir_threshold_db")), snir_threshold_db
        ):
            continue
        if not _matches_optional(
            _float_or_none(row.get("snir_threshold_min_db")), snir_threshold_min_db
        ):
            continue
        if not _matches_optional(
            _float_or_none(row.get("snir_threshold_max_db")), snir_threshold_max_db
        ):
            continue
        filtered.append(row)
    return filtered


def _filter_rows(
    rows: list[dict[str, object]],
    snir_modes: Iterable[str],
    cluster: str,
    *,
    snir_threshold_db: float | None = None,
    snir_threshold_min_db: float | None = None,
    snir_threshold_max_db: float | None = None,
) -> list[dict[str, object]]:
    ensure_network_size(rows)
    normalized_cluster = cluster.strip().lower()
    filtered: list[dict[str, object]] = []
    for row in rows:
        snir_mode = _normalize_snir(row.get("snir_mode"))
        if snir_mode is None or snir_mode not in snir_modes:
            continue
        cluster_value = str(row.get("cluster", "all")).strip().lower()
        if normalized_cluster and cluster_value != normalized_cluster:
            continue
        row["snir_mode"] = snir_mode
        filtered.append(row)
    filtered = filter_mixra_opt_fallback(filtered)
    return _filter_by_snir_threshold(
        filtered,
        snir_threshold_db=snir_threshold_db,
        snir_threshold_min_db=snir_threshold_min_db,
        snir_threshold_max_db=snir_threshold_max_db,
    )


def _resolve_metric_key(
    rows: list[dict[str, object]],
    candidates: Iterable[str],
    label: str,
) -> str:
    for candidate in candidates:
        if any(candidate in row for row in rows):
            return candidate
    raise ValueError(
        f"Aucune colonne {label} trouvée. Colonnes candidates: {', '.join(candidates)}"
    )


def _derive_metric_key(metric_key: str, base: str) -> str:
    for suffix in ("_mean", "_p50", "_p10", "_p90"):
        if metric_key.endswith(suffix):
            return f"{base}{suffix}"
    return f"{base}_mean"


def _add_derived_der(rows: list[dict[str, object]], pdr_key: str) -> str:
    der_key = _derive_metric_key(pdr_key, "der")
    for row in rows:
        value = row.get(pdr_key)
        if isinstance(value, (int, float)):
            row[der_key] = 1.0 - float(value)
    return der_key


def _group_author_curves(
    curves: list[AuthorCurve],
    metric: str,
) -> dict[tuple[str, str | None], list[AuthorCurve]]:
    grouped: dict[tuple[str, str | None], list[AuthorCurve]] = {}
    for curve in curves:
        if curve.metric != metric:
            continue
        grouped.setdefault((curve.snir_mode, curve.algo), []).append(curve)
    return grouped


def _plot_author_overlays(
    ax: plt.Axes,
    curves: list[AuthorCurve],
    metric: str,
) -> None:
    grouped = _group_author_curves(curves, metric)
    for (snir_mode, algo), entries in grouped.items():
        entries = sorted(entries, key=lambda item: item.x)
        x_values = [entry.x for entry in entries]
        y_values = [entry.y for entry in entries]
        label_prefix = "Auteurs"
        color = "#444444"
        if algo:
            label_prefix = f"Auteurs {algo_label(algo)}"
            color = ALGO_COLORS.get(algo, color)
        label = entries[0].label or f"{label_prefix} ({SNIR_LABELS.get(snir_mode, snir_mode)})"
        ax.plot(
            x_values,
            y_values,
            linestyle="--",
            linewidth=1.2,
            marker="x",
            color=color,
            alpha=0.7,
            label=label,
        )


def _render_metric_plot(
    rows: list[dict[str, object]],
    metric_key: str,
    metric_label: str,
    output_stem: str,
    output_dir: Path,
    author_curves: list[AuthorCurve],
    y_limits: tuple[float, float] | None = None,
    *,
    close_figures: bool = True,
    enable_suptitle: bool = True,
) -> None:
    fig, ax = plt.subplots(1, 1)
    status = is_constant_metric(metric_values(rows, metric_key))
    if status is not MetricStatus.OK:
        render_metric_status(fig, ax, status)
    else:
        plot_metric_by_snir(
            ax,
            rows,
            metric_key,
            use_algo_styles=True,
            label_percentiles=True,
        )
        _plot_author_overlays(ax, author_curves, metric_label.lower())
    ax.set_xlabel(label_for("x.network_size"))
    ax.set_ylabel(metric_label)
    if y_limits:
        ax.set_ylim(*y_limits)
    apply_suptitle(
        fig,
        f"{metric_label} vs taille du réseau (SNIR on/off)",
        enable_suptitle=enable_suptitle,
        y=SUPTITLE_Y,
    )
    handles, labels = collect_legend_entries(ax)
    handles, labels = deduplicate_legend_entries(handles, labels)
    if not handles:
        handles, labels = fallback_legend_handles()
    ax.legend(handles, labels, loc="best")
    apply_figure_layout(fig)
    save_figure(fig, output_dir, output_stem)
    if close_figures:
        plt.close(fig)


def _parse_snir_modes(value: str) -> list[str]:
    raw = [item.strip().lower() for item in value.split(",") if item.strip()]
    modes = [mode for mode in raw if mode in SNIR_MODES]
    if not modes:
        raise ValueError("Aucun snir_mode valide (snir_on/snir_off) fourni.")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PDR/DER/throughput entre SNIR on/off à partir des CSV agrégés "
            "Step1/Step2 et superpose les courbes auteurs."
        )
    )
    parser.add_argument(
        "--step1-csv",
        type=Path,
        default=Path("article_c/step1/results/aggregates/aggregated_results.csv"),
        help="Chemin vers aggregated_results.csv de l'étape 1.",
    )
    parser.add_argument(
        "--step2-csv",
        type=Path,
        default=Path("article_c/step2/results/aggregates/aggregated_results.csv"),
        help="Chemin vers aggregated_results.csv de l'étape 2.",
    )
    parser.add_argument(
        "--author-curves",
        type=Path,
        default=Path("article_c/common/data/author_curves_snir.csv"),
        help=(
            "CSV optionnel pour les courbes auteurs (colonnes: metric, snir_mode, x, y, "
            "label?, algo?)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("article_c/plots/output/compare_with_snir"),
        help="Répertoire de sortie des figures.",
    )
    parser.add_argument(
        "--snir-modes",
        type=_parse_snir_modes,
        default=_parse_snir_modes("snir_on,snir_off"),
        help="Liste de modes SNIR (ex: snir_on,snir_off).",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=None,
        help="Filtre les lignes sur un seuil SNIR précis (dB).",
    )
    parser.add_argument(
        "--snir-threshold-min-db",
        type=float,
        default=None,
        help="Filtre les lignes sur la borne basse de clamp SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-max-db",
        type=float,
        default=None,
        help="Filtre les lignes sur la borne haute de clamp SNIR (dB).",
    )
    parser.add_argument(
        "--cluster",
        default="all",
        help="Filtre les lignes sur ce cluster (défaut: all).",
    )
    parser.add_argument(
        "--formats",
        default=None,
        help="Formats d'export séparés par des virgules (ex: png,eps).",
    )
    parser.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Désactive le titre global (suptitle) des figures.",
    )
    return parser


def main(argv: list[str] | None = None, *, close_figures: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    export_formats = parse_export_formats(args.formats)
    set_default_export_formats(export_formats)

    apply_plot_style()

    step1_rows = _load_results_with_fallback(args.step1_csv, step=1)
    step2_rows = _load_results_with_fallback(args.step2_csv, step=2)

    snir_modes = args.snir_modes
    step1_rows = _filter_rows(
        step1_rows,
        snir_modes,
        args.cluster,
        snir_threshold_db=args.snir_threshold_db,
        snir_threshold_min_db=args.snir_threshold_min_db,
        snir_threshold_max_db=args.snir_threshold_max_db,
    )
    step2_rows = _filter_rows(
        step2_rows,
        snir_modes,
        args.cluster,
        snir_threshold_db=args.snir_threshold_db,
        snir_threshold_min_db=args.snir_threshold_min_db,
        snir_threshold_max_db=args.snir_threshold_max_db,
    )

    author_curves = _load_author_curves(args.author_curves)

    pdr_key = _resolve_metric_key(
        step1_rows,
        ("pdr_mean", "pdr_p50", "pdr"),
        "PDR",
    )
    der_key = _add_derived_der(step1_rows, pdr_key)
    throughput_key = _resolve_metric_key(
        step2_rows,
        (
            "throughput_success_mean",
            "throughput_success_p50",
            "throughput_mean",
            "throughput",
        ),
        "throughput",
    )

    enable_suptitle = not args.no_suptitle
    _render_metric_plot(
        step1_rows,
        pdr_key,
        "PDR",
        "compare_pdr_snir",
        args.output_dir,
        author_curves,
        y_limits=(0.0, 1.0),
        close_figures=close_figures,
        enable_suptitle=enable_suptitle,
    )
    _render_metric_plot(
        step1_rows,
        der_key,
        "DER",
        "compare_der_snir",
        args.output_dir,
        author_curves,
        y_limits=(0.0, 1.0),
        close_figures=close_figures,
        enable_suptitle=enable_suptitle,
    )
    _render_metric_plot(
        step2_rows,
        throughput_key,
        "Throughput",
        "compare_throughput_snir",
        args.output_dir,
        author_curves,
        close_figures=close_figures,
        enable_suptitle=enable_suptitle,
    )


if __name__ == "__main__":
    main()
