"""Génère des nuages de points corrélant plusieurs métriques QoS."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from article_c.common.plot_helpers import apply_figure_layout
try:  # pragma: no cover - dépend du mode d'exécution
    from .lfs_metrics import (
        MethodScenarioMetrics,
        cluster_targets_from_config,
        load_all_metrics,
        load_gateway_position,
        load_yaml_config,
    )
except ImportError:  # pragma: no cover - fallback pour exécution directe
    from lfs_metrics import (  # type: ignore
        MethodScenarioMetrics,
        cluster_targets_from_config,
        load_all_metrics,
        load_gateway_position,
        load_yaml_config,
    )


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "*",
    "X",
    "<",
    ">",
]


AXIS_LABELS = {
    "attempted": "Attempted transmissions",
    "collisions": "Collision count",
    "collision_rate": "Collision rate",
    "delivered": "Delivered packets",
    "der_global": "Global DER",
    "energy_j": "Total energy (J)",
    "energy_per_attempt": "Energy per attempt (J/msg)",
    "energy_per_delivery": "Energy per delivery (J/msg)",
    "jain_index": "Jain fairness index",
    "loss_rate": "Packet loss rate",
    "min_sf_share": "Minimum SF share",
    "pdr_global": "Global PDR",
}


FRACTION_METRICS = {
    "collision_rate",
    "der_global",
    "jain_index",
    "loss_rate",
    "min_sf_share",
    "pdr_global",
}


@dataclass
class DataPoint:
    """Représente un point (méthode/scénario) pour le tracé."""

    method: str
    scenario: str
    x: float
    y: float
    color_value: Optional[float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lfs_plots_scatter",
        description=(
            "Crée un scatter plot corrélant deux métriques QoS avec une option de couleur" \
            " pour une troisième dimension."
        ),
    )
    parser.add_argument("--in", dest="root", type=Path, required=True, help="Dossier racine des résultats (<méthode>/<scénario>).")
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        required=False,
        help="Fichier YAML décrivant les scénarios (utile pour les cibles QoS et l'ordre d'affichage).",
    )
    parser.add_argument(
        "--x",
        dest="x_metric",
        required=True,
        help="Nom de la métrique à placer sur l'axe X (ex: pdr_global, energy_per_delivery, pdr_gap_by_cluster:0).",
    )
    parser.add_argument(
        "--y",
        dest="y_metric",
        required=True,
        help="Nom de la métrique à placer sur l'axe Y.",
    )
    parser.add_argument(
        "--color",
        dest="color_metric",
        required=False,
        help="Nom de la métrique utilisée comme échelle de couleur (optionnel).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        dest="methods",
        default=None,
        help="Sous-ensemble de méthodes à afficher (par nom de dossier).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        dest="scenarios",
        default=None,
        help="Sous-ensemble de scénarios à inclure (par identifiant).",
    )
    parser.add_argument(
        "--out",
        dest="output",
        type=Path,
        default=Path("qos_cli") / "figures" / "scatter.png",
        help="Fichier de sortie pour le graphique (PNG).",
    )
    parser.add_argument(
        "--title",
        dest="title",
        default=None,
        help="Titre personnalisé pour le graphique.",
    )
    parser.add_argument(
        "--cmap",
        dest="cmap",
        default="viridis",
        help="Nom de la palette matplotlib pour la métrique de couleur.",
    )
    parser.add_argument(
        "--dpi",
        dest="dpi",
        type=int,
        default=150,
        help="Résolution (dpi) du fichier de sortie (défaut: 150).",
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        action="store_true",
        help="Relie les points d'une même méthode pour obtenir une courbe paramétrique.",
    )
    parser.add_argument(
        "--annotate",
        dest="annotate",
        action="store_true",
        help="Ajoute l'identifiant du scénario à côté de chaque point.",
    )
    parser.add_argument(
        "--size",
        dest="size",
        type=float,
        default=6.0,
        help="Largeur (pouces) du graphique. La hauteur est ajustée proportionnellement.",
    )
    return parser.parse_args(argv)


def _split_metric_spec(spec: str) -> Tuple[str, Optional[str]]:
    base, sep, key = spec.partition(":")
    if sep:
        return base.strip(), key.strip() or None
    return spec.strip(), None


def _metric_label(spec: str) -> str:
    base, key = _split_metric_spec(spec)
    if base == "cluster_pdr" and key is not None:
        return f"Cluster {key} PDR"
    if base == "cluster_targets" and key is not None:
        return f"Cluster {key} target"
    if base == "pdr_gap_by_cluster" and key is not None:
        return f"Cluster {key} PDR gap"
    return AXIS_LABELS.get(base, base)


def _is_fraction_metric(spec: str) -> bool:
    base, key = _split_metric_spec(spec)
    if base in FRACTION_METRICS:
        return True
    if base in {"cluster_pdr", "cluster_targets"}:
        return True
    if base == "pdr_gap_by_cluster" and key is None:
        return False
    return False


def _resolve_mapping_value(mapping: Mapping[str, float], key: Optional[str]) -> Optional[float]:
    if key is None:
        return None
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_metric(metric: MethodScenarioMetrics, spec: str) -> Optional[float]:
    base, key = _split_metric_spec(spec)
    if not base:
        return None

    if not hasattr(metric, base):
        return None
    value = getattr(metric, base)

    if isinstance(value, Mapping):
        resolved = _resolve_mapping_value(value, key)
        return resolved

    if key is not None:
        return None

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gather_points(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
    x_spec: str,
    y_spec: str,
    color_spec: Optional[str],
    allowed_methods: Optional[Iterable[str]],
    allowed_scenarios: Optional[Iterable[str]],
) -> Tuple[List[DataPoint], List[str]]:
    allowed_methods_set = {m for m in allowed_methods} if allowed_methods else None
    allowed_scenarios_set = {s for s in allowed_scenarios} if allowed_scenarios else None

    warnings: List[str] = []
    points: List[DataPoint] = []

    for (method, scenario), metric in metrics.items():
        if allowed_methods_set is not None and method not in allowed_methods_set:
            continue
        if allowed_scenarios_set is not None and scenario not in allowed_scenarios_set:
            continue

        x_value = _resolve_metric(metric, x_spec)
        y_value = _resolve_metric(metric, y_spec)

        if x_value is None or math.isnan(x_value):
            warnings.append(f"Donnée manquante pour '{x_spec}' ({method}/{scenario}).")
            continue
        if y_value is None or math.isnan(y_value):
            warnings.append(f"Donnée manquante pour '{y_spec}' ({method}/{scenario}).")
            continue

        color_value: Optional[float] = None
        if color_spec is not None:
            color_value = _resolve_metric(metric, color_spec)
            if color_value is None or math.isnan(color_value):
                warnings.append(
                    f"Métrique couleur indisponible pour '{color_spec}' ({method}/{scenario})."
                )
                color_value = None

        points.append(DataPoint(method=method, scenario=scenario, x=float(x_value), y=float(y_value), color_value=None if color_value is None else float(color_value)))

    return points, warnings


def _scenario_order(scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]]) -> Dict[str, int]:
    if scenarios_cfg is None:
        return {}
    order: Dict[str, int] = {}
    for index, scenario in enumerate(scenarios_cfg.keys()):
        order[str(scenario)] = index
    return order


def _tolerance_lines(
    spec: str,
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
    orientation: str,
    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]],
) -> List[Tuple[float, str]]:
    base, key = _split_metric_spec(spec)
    lines: List[Tuple[float, str]] = []

    def fmt_percent(value: float) -> str:
        return f"{value * 100:.1f}%"

    if base == "pdr_global":
        targets: List[float] = []
        for metric in metrics.values():
            targets.extend(metric.cluster_targets.values())
        if scenarios_cfg:
            for scenario in scenarios_cfg.values():
                targets.extend(cluster_targets_from_config(scenario).values())
        targets = [float(t) for t in targets if not math.isnan(float(t))]
        if targets:
            threshold = min(targets)
            lines.append((threshold, f"Min target {fmt_percent(threshold)}"))
    elif base == "cluster_pdr" and key:
        targets = []
        for metric in metrics.values():
            value = metric.cluster_targets.get(key)
            if value is not None:
                targets.append(float(value))
        if scenarios_cfg:
            for scenario in scenarios_cfg.values():
                target = cluster_targets_from_config(scenario).get(key)
                if target is not None:
                    targets.append(float(target))
        if targets:
            target_value = min(targets)
            lines.append((target_value, f"Cluster {key} target {fmt_percent(target_value)}"))
    elif base == "pdr_gap_by_cluster":
        lines.append((0.0, "Gap = 0"))
    elif base == "collision_rate":
        lines.append((0.05, "5% collision rate"))
    elif base == "loss_rate":
        lines.append((0.1, "10% loss"))

    if orientation == "v":
        # Pour les lignes verticales la valeur représente X.
        return lines
    return lines


def _apply_tolerance_lines(ax: plt.Axes, lines: List[Tuple[float, str]], orientation: str) -> None:
    added_labels: set[str] = set()
    for value, label in lines:
        if orientation == "h":
            handle = ax.axhline(value, color="red", linestyle="--", linewidth=1.0)
        else:
            handle = ax.axvline(value, color="red", linestyle="--", linewidth=1.0)
        if label not in added_labels:
            handle.set_label(label)
            added_labels.add(label)


def _plot_points(
    points: List[DataPoint],
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
    x_spec: str,
    y_spec: str,
    color_spec: Optional[str],
    cmap: str,
    connect: bool,
    annotate: bool,
    title: Optional[str],
    output: Path,
    dpi: int,
    width: float,
    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]],
) -> None:
    if not points:
        raise RuntimeError("Impossible de tracer le graphique : aucune donnée exploitable.")

    order_lookup = _scenario_order(scenarios_cfg)

    points_by_method: Dict[str, List[DataPoint]] = {}
    for point in points:
        points_by_method.setdefault(point.method, []).append(point)

    fig, ax = plt.subplots(figsize=(width, max(4.0, width * 0.7)))

    color_values = [point.color_value for point in points if point.color_value is not None]
    norm: Optional[mcolors.Normalize] = None
    if color_spec and color_values:
        norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))

    for index, (method, method_points) in enumerate(sorted(points_by_method.items())):
        x_vals = [point.x for point in method_points]
        y_vals = [point.y for point in method_points]
        markers = MARKERS[index % len(MARKERS)]
        if color_spec and norm is not None:
            colors = [point.color_value if point.color_value is not None else float("nan") for point in method_points]
            scatter = ax.scatter(x_vals, y_vals, c=colors, cmap=cmap, norm=norm, marker=markers, edgecolor="black", linewidths=0.5, label=method)
        else:
            color = COLORS[index % len(COLORS)]
            scatter = ax.scatter(x_vals, y_vals, color=color, marker=markers, edgecolor="black", linewidths=0.5, label=method)

        if connect and len(method_points) >= 2:
            sorted_points = sorted(
                method_points,
                key=lambda item: (
                    order_lookup.get(item.scenario, math.inf),
                    item.scenario,
                ),
            )
            ax.plot(
                [point.x for point in sorted_points],
                [point.y for point in sorted_points],
                color=COLORS[index % len(COLORS)],
                linewidth=1.0,
                alpha=0.7,
            )

        if annotate:
            for point in method_points:
                ax.annotate(
                    point.scenario,
                    (point.x, point.y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    ax.set_xlabel(_metric_label(x_spec))
    ax.set_ylabel(_metric_label(y_spec))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(title="Method", loc="best")

    if color_spec and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(_metric_label(color_spec))

    if _is_fraction_metric(x_spec):
        ax.set_xlim(0.0, 1.0)
    if _is_fraction_metric(y_spec):
        ax.set_ylim(0.0, 1.0)

    tolerance_y = _tolerance_lines(y_spec, metrics, "h", scenarios_cfg)
    if tolerance_y:
        _apply_tolerance_lines(ax, tolerance_y, "h")
    tolerance_x = _tolerance_lines(x_spec, metrics, "v", scenarios_cfg)
    if tolerance_x:
        _apply_tolerance_lines(ax, tolerance_x, "v")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{_metric_label(y_spec)} vs {_metric_label(x_spec)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_layout(fig, tight_layout=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if not args.root.exists():
        raise FileNotFoundError(f"Dossier de résultats introuvable : {args.root}")

    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]] = None
    gateway_position: Optional[Tuple[float, float]] = None
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {args.config}")
        scenarios_cfg = load_yaml_config(args.config)
        gateway_position = load_gateway_position(args.config)

    all_metrics = load_all_metrics(args.root, scenarios_cfg, gateway_position=gateway_position)
    if not all_metrics:
        raise RuntimeError("Aucune métrique détectée – vérifiez la structure des résultats.")

    points, warnings = _gather_points(
        all_metrics,
        args.x_metric,
        args.y_metric,
        args.color_metric,
        args.methods,
        args.scenarios,
    )

    _plot_points(
        points,
        all_metrics,
        args.x_metric,
        args.y_metric,
        args.color_metric,
        args.cmap,
        args.connect,
        args.annotate,
        args.title,
        args.output,
        args.dpi,
        args.size,
        scenarios_cfg,
    )

    for message in warnings:
        print(f"[WARN] {message}")


if __name__ == "__main__":  # pragma: no cover - exécution CLI
    main()
