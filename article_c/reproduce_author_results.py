"""Reproduit les figures 4, 5, 7 et 8 de l'article QoS.

Le script charge les résultats agrégés LoRaFlexSim (step1/step2) et superpose,
si disponibles, les courbes des auteurs à partir d'un CSV dédié.

Optionnellement, il peut exporter les points tracés sous forme de CSV structurés
pour faciliter la vérification des figures.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from plot_defaults import resolve_ieee_figsize
from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    ALGO_COLORS,
    MetricStatus,
    add_global_legend,
    apply_suptitle,
    apply_figure_layout,
    apply_plot_style,
    collect_legend_entries,
    deduplicate_legend_entries,
    filter_mixra_opt_fallback,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    load_step2_aggregated,
    metric_values,
    parse_export_formats,
    render_metric_status,
    save_figure,
    select_received_metric_key,
    set_default_figure_clamp_enabled,
    set_default_export_formats,
    set_network_size_ticks,
)
from article_c.common.plotting_style import (
    SUPTITLE_Y,
    apply_output_fonttype,
)

LOGGER = logging.getLogger(__name__)

EXPORT_FIELDS = (
    "figure",
    "metric",
    "profile",
    "cluster",
    "source",
    "x",
    "x_label",
    "y",
    "label",
)


@dataclass(frozen=True)
class QoSProfile:
    key: str
    label: str
    algo_aliases: tuple[str, ...]
    pdr_targets: dict[str, float]
    color: str


DEFAULT_PDR_TARGETS = (0.9, 0.8, 0.7)

QOS_PROFILES: dict[str, QoSProfile] = {
    "mixra": QoSProfile(
        key="mixra",
        label="MixRA",
        algo_aliases=("mixra", "mixra_opt", "mixra_h"),
        pdr_targets={
            cluster: DEFAULT_PDR_TARGETS[idx]
            for idx, cluster in enumerate(DEFAULT_CONFIG.qos.clusters)
        },
        color=ALGO_COLORS.get("mixra_opt", "#2ca02c"),
    ),
    "apra": QoSProfile(
        key="apra",
        label="APRA",
        algo_aliases=("apra",),
        pdr_targets={
            cluster: DEFAULT_PDR_TARGETS[idx]
            for idx, cluster in enumerate(DEFAULT_CONFIG.qos.clusters)
        },
        color="#1f77b4",
    ),
    "aimi": QoSProfile(
        key="aimi",
        label="Aimi",
        algo_aliases=("aimi",),
        pdr_targets={
            cluster: DEFAULT_PDR_TARGETS[idx]
            for idx, cluster in enumerate(DEFAULT_CONFIG.qos.clusters)
        },
        color="#ff7f0e",
    ),
}

TRAFFIC_LOAD_LEVELS = (
    ("faible", 0.85),
    ("moyenne", 1.15),
    ("élevée", 1.45),
    ("très élevée", math.inf),
)


@dataclass(frozen=True)
class AuthorCurve:
    figure: int
    profile: str
    cluster: str
    x: float
    y: float
    label: str


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


def _normalize_algo(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _select_profiles(keys: Iterable[str]) -> list[QoSProfile]:
    profiles: list[QoSProfile] = []
    for key in keys:
        normalized = key.strip().lower()
        profile = QOS_PROFILES.get(normalized)
        if profile is None:
            raise ValueError(f"Profil QoS inconnu: {key}")
        profiles.append(profile)
    return profiles


def _load_author_curves(path: Path) -> list[AuthorCurve]:
    if not path.exists():
        LOGGER.info("Aucun fichier de courbes auteurs: %s", path)
        return []
    curves: list[AuthorCurve] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            try:
                figure = int(str(row.get("figure", "")).strip())
            except ValueError:
                continue
            profile = str(row.get("profile", "")).strip().lower()
            cluster = str(row.get("cluster", "")).strip().lower()
            try:
                x_value = float(row.get("x", ""))
                y_value = float(row.get("y", ""))
            except (TypeError, ValueError):
                continue
            label = str(row.get("label", "")).strip()
            curves.append(
                AuthorCurve(
                    figure=figure,
                    profile=profile,
                    cluster=cluster,
                    x=x_value,
                    y=y_value,
                    label=label,
                )
            )
    return curves


def _group_author_curves(
    curves: list[AuthorCurve],
    figure: int,
    profile: str,
    cluster: str,
) -> list[AuthorCurve]:
    return [
        curve
        for curve in curves
        if curve.figure == figure
        and curve.profile == profile
        and curve.cluster == cluster
    ]


def _plot_author_overlay(
    ax: plt.Axes,
    curves: list[AuthorCurve],
    label_prefix: str,
    color: str,
) -> None:
    if not curves:
        return
    curves = sorted(curves, key=lambda item: item.x)
    x_values = [curve.x for curve in curves]
    y_values = [curve.y for curve in curves]
    label = curves[0].label or f"{label_prefix} (auteurs)"
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


def _append_export_row(
    export_rows: list[dict[str, object]] | None,
    *,
    figure: int,
    metric: str,
    profile: str,
    cluster: str,
    source: str,
    x: float,
    y: float,
    label: str,
    x_label: str | None = None,
) -> None:
    if export_rows is None:
        return
    export_rows.append(
        {
            "figure": figure,
            "metric": metric,
            "profile": profile,
            "cluster": cluster,
            "source": source,
            "x": x,
            "x_label": x_label or "",
            "y": y,
            "label": label,
        }
    )


def _export_author_curves(
    export_rows: list[dict[str, object]] | None,
    *,
    curves: list[AuthorCurve],
    figure: int,
    metric: str,
    profile: str,
    cluster: str,
    label_prefix: str,
) -> None:
    if export_rows is None or not curves:
        return
    label = curves[0].label or f"{label_prefix} (auteurs)"
    for curve in curves:
        _append_export_row(
            export_rows,
            figure=figure,
            metric=metric,
            profile=profile,
            cluster=cluster,
            source="author",
            x=curve.x,
            y=curve.y,
            label=label,
        )


def _write_plot_csv(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        LOGGER.info("Aucune donnée CSV à écrire pour %s.", stem)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("CSV exporté: %s", path)


def _resolve_snir_mode(rows: list[dict[str, object]]) -> str | None:
    modes = {str(row.get("snir_mode", "")) for row in rows if row.get("snir_mode")}
    if "snir_on" in modes:
        return "snir_on"
    return next(iter(modes), None)


def _filter_rows_by_profile(
    rows: list[dict[str, object]],
    profile: QoSProfile,
) -> list[dict[str, object]]:
    aliases = {alias.lower() for alias in profile.algo_aliases}
    return [
        row
        for row in rows
        if _normalize_algo(row.get("algo")) in aliases
    ]


def _filter_cluster_rows(
    rows: list[dict[str, object]],
    cluster: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("cluster", "all")).lower() == cluster
    ]


def _collect_metric_points(
    rows: list[dict[str, object]],
    metric_key: str,
) -> dict[int, float]:
    points: dict[int, float] = {}
    for row in rows:
        value = row.get(metric_key)
        size = row.get("network_size")
        if not isinstance(size, (int, float)) or not isinstance(value, (int, float)):
            continue
        points[int(round(size))] = float(value)
    return points


def _render_legend(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    axes_list = list(axes)
    handles, labels = collect_legend_entries(axes_list)
    handles, labels = deduplicate_legend_entries(handles, labels)
    add_global_legend(
        fig,
        axes_list,
        legend_loc="above",
        handles=handles,
        labels=labels,
    )


def _finalize_figure(fig: plt.Figure, title: str, *, show_header: bool = True) -> None:
    apply_suptitle(fig, title, enable_suptitle=show_header, y=SUPTITLE_Y)
    apply_figure_layout(
        fig,
        figsize=resolve_ieee_figsize(len(fig.axes)),
        margins=legend_margins("above"),
        figure_clamp=False,
    )


def plot_fig4(
    rows: list[dict[str, object]],
    profiles: list[QoSProfile],
    author_curves: list[AuthorCurve],
    export_rows: list[dict[str, object]] | None = None,
    *,
    show_header: bool = True,
) -> plt.Figure | None:
    cluster_names = [
        cluster
        for cluster in DEFAULT_CONFIG.qos.clusters
        if any(str(row.get("cluster", "")).lower() == cluster for row in rows)
    ]
    if not cluster_names:
        LOGGER.warning("Aucune donnée cluster trouvée pour la Fig.4.")
        return None
    metric_status = is_constant_metric(metric_values(rows, "pdr_mean"))
    fig, axes = plt.subplots(1, len(cluster_names), sharey=True)
    if len(cluster_names) == 1:
        axes = [axes]
    if metric_status is not MetricStatus.OK:
        render_metric_status(fig, axes, metric_status)
        _finalize_figure(
            fig,
            "Fig.4 - DER par cluster vs taille du réseau",
            show_header=show_header,
        )
        return fig

    for ax, cluster in zip(axes, cluster_names, strict=False):
        cluster_rows = _filter_cluster_rows(rows, cluster)
        for profile in profiles:
            profile_label = profile.label or profile.key
            simulation_label = f"{profile_label} (LoRaFlexSim)"
            profile_rows = _filter_rows_by_profile(cluster_rows, profile)
            points = _collect_metric_points(profile_rows, "pdr_mean")
            if not points:
                continue
            sizes = sorted(points)
            der_values = [1.0 - points[size] for size in sizes]
            ax.plot(
                sizes,
                der_values,
                marker="o",
                label=simulation_label,
                color=profile.color,
            )
            for size, value in zip(sizes, der_values, strict=False):
                _append_export_row(
                    export_rows,
                    figure=4,
                    metric="der",
                    profile=profile.key,
                    cluster=cluster,
                    source="loraflexsim",
                    x=float(size),
                    y=float(value),
                    label=simulation_label,
                )
            overlay = _group_author_curves(
                author_curves,
                figure=4,
                profile=profile.key,
                cluster=cluster,
            )
            _plot_author_overlay(ax, overlay, profile_label, profile.color)
            _export_author_curves(
                export_rows,
                curves=overlay,
                figure=4,
                metric="der",
                profile=profile.key,
                cluster=cluster,
                label_prefix=profile_label,
            )
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel("DER")
        ax.set_ylim(0.0, 1.0)
        set_network_size_ticks(ax, sorted({int(row["network_size"]) for row in cluster_rows}))

    _render_legend(fig, axes)
    _finalize_figure(
        fig,
        "Fig.4 - DER par cluster vs taille du réseau",
        show_header=show_header,
    )
    return fig


def _resolve_load_key(rows: list[dict[str, object]]) -> str | None:
    for candidate in (
        "traffic_coeff_mean",
        "traffic_coeff_p50",
        "traffic_coeff",
        "network_load_mean",
        "network_load",
        "load",
    ):
        if any(candidate in row for row in rows):
            return candidate
    return None


def _load_level(value: float) -> str:
    for label, threshold in TRAFFIC_LOAD_LEVELS:
        if value <= threshold:
            return label
    return TRAFFIC_LOAD_LEVELS[-1][0]


def plot_fig5(
    rows: list[dict[str, object]],
    profiles: list[QoSProfile],
    author_curves: list[AuthorCurve],
    export_rows: list[dict[str, object]] | None = None,
    *,
    show_header: bool = True,
) -> plt.Figure | None:
    load_key = _resolve_load_key(rows)
    if load_key is None:
        LOGGER.warning("Aucune colonne de charge trouvée pour la Fig.5.")
        return None
    metric_status = is_constant_metric(metric_values(rows, "pdr_mean"))
    fig, ax = plt.subplots(1, 1)
    if metric_status is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_status)
        _finalize_figure(
            fig,
            "Fig.5 - DER selon la charge",
            show_header=show_header,
        )
        return fig

    load_labels = [label for label, _ in TRAFFIC_LOAD_LEVELS]
    x_positions = list(range(len(load_labels)))

    for profile in profiles:
        profile_label = profile.label or profile.key
        simulation_label = f"{profile_label} (LoRaFlexSim)"
        profile_rows = _filter_rows_by_profile(rows, profile)
        aggregated: dict[str, list[float]] = {label: [] for label in load_labels}
        for row in profile_rows:
            load_value = row.get(load_key)
            pdr_value = row.get("pdr_mean")
            if not isinstance(load_value, (int, float)) or not isinstance(
                pdr_value, (int, float)
            ):
                continue
            label = _load_level(float(load_value))
            aggregated[label].append(1.0 - float(pdr_value))
        y_values = [
            sum(aggregated[label]) / len(aggregated[label])
            if aggregated[label]
            else float("nan")
            for label in load_labels
        ]
        ax.plot(
            x_positions,
            y_values,
            marker="o",
            label=simulation_label,
            color=profile.color,
        )
        for x_value, y_value, label in zip(
            x_positions,
            y_values,
            load_labels,
            strict=False,
        ):
            if math.isnan(y_value):
                continue
            _append_export_row(
                export_rows,
                figure=5,
                metric="der",
                profile=profile.key,
                cluster="all",
                source="loraflexsim",
                x=float(x_value),
                x_label=label,
                y=float(y_value),
                label=simulation_label,
            )
        overlay = _group_author_curves(
            author_curves,
            figure=5,
            profile=profile.key,
            cluster="all",
        )
        _plot_author_overlay(ax, overlay, profile_label, profile.color)
        _export_author_curves(
            export_rows,
            curves=overlay,
            figure=5,
            metric="der",
            profile=profile.key,
            cluster="all",
            label_prefix=profile_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.capitalize() for label in load_labels])
    ax.set_xlabel("Charge réseau")
    ax.set_ylabel("DER")
    ax.set_ylim(0.0, 1.0)
    _render_legend(fig, [ax])
    _finalize_figure(fig, "Fig.5 - DER selon la charge", show_header=show_header)
    return fig


def plot_fig7(
    rows: list[dict[str, object]],
    profiles: list[QoSProfile],
    author_curves: list[AuthorCurve],
    export_rows: list[dict[str, object]] | None = None,
    *,
    show_header: bool = True,
) -> plt.Figure | None:
    if not rows:
        LOGGER.warning("Aucune donnée disponible pour la Fig.7.")
        return None
    metric_status = is_constant_metric(metric_values(rows, "sent_mean"))
    fig, ax = plt.subplots(1, 1)
    if metric_status is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_status)
        _finalize_figure(
            fig,
            "Fig.7 - Sacrifice d'offre de trafic",
            show_header=show_header,
        )
        return fig

    sizes = sorted(
        {
            int(row["network_size"])
            for row in rows
            if isinstance(row.get("network_size"), (int, float))
        }
    )
    if not sizes:
        LOGGER.warning("Tailles de réseau absentes pour la Fig.7.")
        return None

    baseline_by_size: dict[int, float] = {}
    for size in sizes:
        sent_values = [
            row.get("sent_mean")
            for row in rows
            if int(row.get("network_size", 0)) == size
            and isinstance(row.get("sent_mean"), (int, float))
        ]
        if sent_values:
            baseline_by_size[size] = max(sent_values)

    for profile in profiles:
        profile_label = profile.label or profile.key
        simulation_label = f"{profile_label} (LoRaFlexSim)"
        profile_rows = _filter_rows_by_profile(rows, profile)
        sacrifices: list[float] = []
        for size in sizes:
            values = [
                row.get("sent_mean")
                for row in profile_rows
                if int(row.get("network_size", 0)) == size
                and isinstance(row.get("sent_mean"), (int, float))
            ]
            if not values or size not in baseline_by_size:
                sacrifices.append(float("nan"))
                continue
            baseline = baseline_by_size[size]
            mean_sent = sum(values) / len(values)
            sacrifices.append(1.0 - (mean_sent / baseline if baseline else 0.0))
        ax.plot(
            sizes,
            sacrifices,
            marker="o",
            label=simulation_label,
            color=profile.color,
        )
        for size, value in zip(sizes, sacrifices, strict=False):
            if math.isnan(value):
                continue
            _append_export_row(
                export_rows,
                figure=7,
                metric="sacrifice",
                profile=profile.key,
                cluster="all",
                source="loraflexsim",
                x=float(size),
                y=float(value),
                label=simulation_label,
            )
        overlay = _group_author_curves(
            author_curves,
            figure=7,
            profile=profile.key,
            cluster="all",
        )
        _plot_author_overlay(ax, overlay, profile_label, profile.color)
        _export_author_curves(
            export_rows,
            curves=overlay,
            figure=7,
            metric="sacrifice",
            profile=profile.key,
            cluster="all",
            label_prefix=profile_label,
        )

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("Sacrifice de trafic (ratio)")
    ax.set_ylim(0.0, 1.0)
    set_network_size_ticks(ax, sizes)
    _render_legend(fig, [ax])
    _finalize_figure(
        fig,
        "Fig.7 - Sacrifice d'offre de trafic",
        show_header=show_header,
    )
    return fig


def plot_fig8(
    rows: list[dict[str, object]],
    profiles: list[QoSProfile],
    author_curves: list[AuthorCurve],
    export_rows: list[dict[str, object]] | None = None,
    *,
    show_header: bool = True,
) -> plt.Figure | None:
    if not rows:
        LOGGER.warning("Aucune donnée disponible pour la Fig.8.")
        return None
    received_key = select_received_metric_key(rows, "received")
    metric_status = is_constant_metric(metric_values(rows, received_key))
    clusters = [
        cluster
        for cluster in (*DEFAULT_CONFIG.qos.clusters, "all")
        if any(str(row.get("cluster", "")).lower() == cluster for row in rows)
    ]
    if not clusters:
        LOGGER.warning("Clusters absents pour la Fig.8.")
        return None

    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    if len(clusters) == 1:
        axes = [axes]
    if metric_status is not MetricStatus.OK:
        render_metric_status(fig, axes, metric_status)
        _finalize_figure(
            fig,
            "Fig.8 - Throughput par cluster",
            show_header=show_header,
        )
        return fig

    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = _filter_cluster_rows(rows, cluster)
        for profile in profiles:
            profile_label = profile.label or profile.key
            simulation_label = f"{profile_label} (LoRaFlexSim)"
            profile_rows = _filter_rows_by_profile(cluster_rows, profile)
            points = _collect_metric_points(profile_rows, received_key)
            if not points:
                continue
            sizes = sorted(points)
            values = [points[size] for size in sizes]
            ax.plot(
                sizes,
                values,
                marker="o",
                label=simulation_label,
                color=profile.color,
            )
            for size, value in zip(sizes, values, strict=False):
                _append_export_row(
                    export_rows,
                    figure=8,
                    metric="throughput",
                    profile=profile.key,
                    cluster=cluster,
                    source="loraflexsim",
                    x=float(size),
                    y=float(value),
                    label=simulation_label,
                )
            overlay = _group_author_curves(
                author_curves,
                figure=8,
                profile=profile.key,
                cluster=cluster,
            )
            _plot_author_overlay(ax, overlay, profile_label, profile.color)
            _export_author_curves(
                export_rows,
                curves=overlay,
                figure=8,
                metric="throughput",
                profile=profile.key,
                cluster=cluster,
                label_prefix=profile_label,
            )
        ax.set_title("Global" if cluster == "all" else f"Cluster {cluster}")
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel("Throughput (paquets reçus)")
        sizes = sorted(
            {
                int(row["network_size"])
                for row in cluster_rows
                if isinstance(row.get("network_size"), (int, float))
            }
        )
        if sizes:
            set_network_size_ticks(ax, sizes)

    _render_legend(fig, axes)
    _finalize_figure(
        fig,
        "Fig.8 - Throughput par cluster",
        show_header=show_header,
    )
    return fig


def _load_results(path: Path, step: int) -> list[dict[str, object]]:
    try:
        if step == 1:
            return load_step1_aggregated(path, allow_sample=False)
        return load_step2_aggregated(path, allow_sample=False)
    except Exception as exc:  # noqa: BLE001 - utile pour rapporter l'absence de CSV
        LOGGER.warning("Impossible de charger %s: %s", path, exc)
        return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduit les figures 4, 5, 7 et 8 de l'article QoS."
    )
    parser.add_argument(
        "--step1-results",
        type=Path,
        default=Path("article_c/step1/results/aggregated_results.csv"),
        help="Chemin vers aggregated_results.csv (step1).",
    )
    parser.add_argument(
        "--step2-results",
        type=Path,
        default=Path("article_c/step2/results/aggregated_results.csv"),
        help="Chemin vers aggregated_results.csv (step2).",
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
        "--output-dir",
        type=Path,
        default=Path("article_c/plots/output"),
        help="Dossier de sortie des figures.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exporte les points des figures (CSV structurés).",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie pour les CSV (défaut: output-dir).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Formats d'export (ex: png,eps).",
    )
    parser.add_argument(
        "--no-figure-clamp",
        action="store_true",
        help="Désactive le clamp de taille des figures.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="mixra,apra,aimi",
        help="Profils QoS à tracer (liste séparée par des virgules).",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="4,5,7,8",
        help="Figures à générer (ex: 4,5,7,8).",
    )
    parser.add_argument(
        "--author-curves",
        type=Path,
        default=Path("article_c/common/data/author_curves.csv"),
        help="CSV contenant les courbes auteurs.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Désactive le titre global (suptitle) des figures.",
    )
    return parser


def main(argv: list[str] | None = None, *, close_figures: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    profiles = _select_profiles(args.profiles.split(","))
    figures = {int(fig.strip()) for fig in args.figures.split(",") if fig.strip()}
    formats = parse_export_formats(args.formats)
    set_default_export_formats(formats)
    set_default_figure_clamp_enabled(not args.no_figure_clamp)
    apply_output_fonttype()
    apply_plot_style()

    step1_rows = _load_results(args.step1_results, step=1)
    step2_rows = _load_results(args.step2_results, step=2)
    if step1_rows:
        step1_rows = filter_mixra_opt_fallback(step1_rows)
        step1_rows = _filter_by_snir_threshold(
            step1_rows,
            snir_threshold_db=args.snir_threshold_db,
            snir_threshold_min_db=args.snir_threshold_min_db,
            snir_threshold_max_db=args.snir_threshold_max_db,
        )
    if step2_rows:
        step2_rows = filter_mixra_opt_fallback(step2_rows)
        step2_rows = _filter_by_snir_threshold(
            step2_rows,
            snir_threshold_db=args.snir_threshold_db,
            snir_threshold_min_db=args.snir_threshold_min_db,
            snir_threshold_max_db=args.snir_threshold_max_db,
        )

    author_curves = _load_author_curves(args.author_curves)
    export_dir = args.csv_output_dir or args.output_dir
    export_data: dict[int, list[dict[str, object]]] = {}

    if 4 in figures:
        export_rows = [] if args.export_csv else None
        fig4 = plot_fig4(
            step1_rows,
            profiles,
            author_curves,
            export_rows,
            show_header=not args.no_header,
        )
        if fig4 is not None:
            save_figure(fig4, args.output_dir, "fig4_der_by_cluster")
            if close_figures:
                plt.close(fig4)
        if args.export_csv and export_rows is not None:
            export_data[4] = export_rows
    if 5 in figures:
        export_rows = [] if args.export_csv else None
        fig5 = plot_fig5(
            step2_rows or step1_rows,
            profiles,
            author_curves,
            export_rows,
            show_header=not args.no_header,
        )
        if fig5 is not None:
            save_figure(fig5, args.output_dir, "fig5_der_by_load")
            if close_figures:
                plt.close(fig5)
        if args.export_csv and export_rows is not None:
            export_data[5] = export_rows
    if 7 in figures:
        export_rows = [] if args.export_csv else None
        fig7 = plot_fig7(
            step1_rows,
            profiles,
            author_curves,
            export_rows,
            show_header=not args.no_header,
        )
        if fig7 is not None:
            save_figure(fig7, args.output_dir, "fig7_traffic_sacrifice")
            if close_figures:
                plt.close(fig7)
        if args.export_csv and export_rows is not None:
            export_data[7] = export_rows
    if 8 in figures:
        export_rows = [] if args.export_csv else None
        fig8 = plot_fig8(
            step1_rows,
            profiles,
            author_curves,
            export_rows,
            show_header=not args.no_header,
        )
        if fig8 is not None:
            save_figure(fig8, args.output_dir, "fig8_throughput_clusters")
            if close_figures:
                plt.close(fig8)
        if args.export_csv and export_rows is not None:
            export_data[8] = export_rows

    if args.export_csv:
        stems = {
            4: "fig4_der_by_cluster",
            5: "fig5_der_by_load",
            7: "fig7_traffic_sacrifice",
            8: "fig8_throughput_clusters",
        }
        for figure, rows in export_data.items():
            _write_plot_csv(export_dir, stems[figure], rows)


if __name__ == "__main__":
    main()
