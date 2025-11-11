"""Génère des cartes de chaleur des métriques QoS en fonction de N et de la période."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - dépend du mode d'exécution
    from .lfs_metrics import (
        MethodScenarioMetrics,
        load_all_metrics,
        load_yaml_config,
    )
except ImportError:  # pragma: no cover - fallback pour exécution directe
    from lfs_metrics import (  # type: ignore
        MethodScenarioMetrics,
        load_all_metrics,
        load_yaml_config,
    )
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit l'espace de noms des arguments pour la CLI."""

    parser = argparse.ArgumentParser(
        prog="lfs_plots_surfaces",
        description=(
            "Trace des heatmaps (PDR global, DER, écart aux cibles) en fonction du nombre de nœuds "
            "et de la période des scénarios QoS."
        ),
    )
    parser.add_argument(
        "--in",
        dest="root",
        type=Path,
        required=True,
        help="Dossier racine contenant les résultats agrégés (<méthode>/<scénario>).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        required=True,
        help="Fichier YAML décrivant les scénarios (utilisé pour récupérer N et la période).",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=Path("qos_cli") / "figures",
        help="Dossier de destination des figures (défaut : qos_cli/figures).",
    )
    return parser.parse_args(argv)


def _group_metrics_by_method(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics]
) -> Dict[str, Dict[str, MethodScenarioMetrics]]:
    grouped: Dict[str, Dict[str, MethodScenarioMetrics]] = {}
    for (method, scenario), data in metrics.items():
        grouped.setdefault(method, {})[scenario] = data
    return grouped


def _sanitize_method_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "method"


def _scenario_parameters(
    scenarios_cfg: Mapping[str, Mapping[str, object]],
    available_scenarios: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    for scenario in available_scenarios:
        cfg = scenarios_cfg.get(scenario)
        if not isinstance(cfg, Mapping):
            continue
        scenario_params: Dict[str, float] = {}
        for key in ("N", "period", "density", "node_density", "gateway_density"):
            value = cfg.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                scenario_params[key] = float(value)
        if "N" in scenario_params and "period" in scenario_params:
            params[scenario] = scenario_params
    return params


def _cluster_targets_from_config(scenario_cfg: Mapping[str, object]) -> Dict[str, float]:
    evaluation = scenario_cfg.get("evaluation") if isinstance(scenario_cfg, Mapping) else None
    cluster_targets: MutableMapping[str, float] = {}
    if isinstance(evaluation, Mapping):
        raw_targets = evaluation.get("cluster_targets")
        if isinstance(raw_targets, Mapping):
            for cluster, value in raw_targets.items():
                if isinstance(value, (int, float)):
                    cluster_targets[str(cluster)] = float(value)
    clusters = scenario_cfg.get("clusters") if isinstance(scenario_cfg, Mapping) else None
    if isinstance(clusters, Mapping):
        for cluster, cluster_cfg in clusters.items():
            if not isinstance(cluster_cfg, Mapping):
                continue
            target = cluster_cfg.get("target_pdr")
            if isinstance(target, (int, float)):
                cluster_targets.setdefault(str(cluster), float(target))
    return dict(cluster_targets)


def _target_gap(
    metrics: MethodScenarioMetrics,
    scenario_cfg: Optional[Mapping[str, object]],
) -> float:
    if scenario_cfg is None:
        return float("nan")
    cluster_targets = _cluster_targets_from_config(scenario_cfg)
    if not cluster_targets:
        return float("nan")
    gaps: List[float] = []
    for cluster, target in cluster_targets.items():
        actual = metrics.cluster_pdr.get(str(cluster))
        if actual is None:
            continue
        gaps.append(float(actual) - target)
    if not gaps:
        return float("nan")
    return min(gaps)


def _metric_value(
    metrics: MethodScenarioMetrics,
    scenario_cfg: Optional[Mapping[str, object]],
    metric: str,
) -> float:
    if metric == "pdr_global":
        return float(metrics.pdr_global) if metrics.pdr_global is not None else float("nan")
    if metric == "der_global":
        return float(metrics.der_global) if metrics.der_global is not None else float("nan")
    if metric == "target_gap":
        return _target_gap(metrics, scenario_cfg)
    raise ValueError(f"Unsupported metric '{metric}'")


def _build_matrix(
    metrics_by_scenario: Mapping[str, MethodScenarioMetrics],
    scenarios_cfg: Mapping[str, Mapping[str, object]],
    scenario_params: Mapping[str, Dict[str, float]],
    n_values: Sequence[float],
    period_values: Sequence[float],
    metric: str,
) -> np.ndarray:
    matrix = np.full((len(n_values), len(period_values)), np.nan, dtype=float)
    index_n = {value: idx for idx, value in enumerate(n_values)}
    index_period = {value: idx for idx, value in enumerate(period_values)}
    for scenario, params in scenario_params.items():
        if "N" not in params or "period" not in params:
            continue
        metrics = metrics_by_scenario.get(scenario)
        if metrics is None:
            continue
        scenario_cfg = scenarios_cfg.get(scenario)
        value = _metric_value(metrics, scenario_cfg, metric)
        row = index_n[params["N"]]
        col = index_period[params["period"]]
        matrix[row, col] = value
    return matrix


def _figure_size(n_count: int, period_count: int) -> Tuple[float, float]:
    width = max(6.0, 1.2 * period_count)
    height = max(4.0, 1.2 * n_count)
    return width, height


def _plot_heatmap(
    matrix: np.ndarray,
    n_values: Sequence[float],
    period_values: Sequence[float],
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=_figure_size(len(n_values), len(period_values)))
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks(range(len(period_values)))
        ax.set_xticklabels([str(int(value)) if value.is_integer() else f"{value:g}" for value in period_values])
        ax.set_yticks(range(len(n_values)))
        ax.set_yticklabels([str(int(value)) if value.is_integer() else f"{value:g}" for value in n_values])
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Node count")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    x_labels = [str(int(value)) if float(value).is_integer() else f"{value:g}" for value in period_values]
    y_labels = [str(int(value)) if float(value).is_integer() else f"{value:g}" for value in n_values]
    ax.set_xticks(range(len(period_values)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Node count")
    ax.set_title(title)

    mask = np.ma.masked_invalid(matrix)
    if mask.count() >= 4:
        try:
            y_coords = np.arange(len(n_values))
            x_coords = np.arange(len(period_values))
            X, Y = np.meshgrid(x_coords, y_coords)
            finite_mask = np.isfinite(matrix)
            if finite_mask.sum() >= 4:
                contour_levels = np.linspace(np.nanmin(matrix), np.nanmax(matrix), num=6)
                if np.nanmin(matrix) != np.nanmax(matrix):
                    cs = ax.contour(
                        X,
                        Y,
                        matrix,
                        levels=contour_levels,
                        colors="black",
                        linewidths=0.5,
                    )
                    ax.clabel(cs, fmt="%.2f", inline=True, fontsize=8)
        except Exception:  # pragma: no cover - contour peut échouer sur des données dégénérées
            pass

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    scenarios_cfg = load_yaml_config(args.config)
    all_metrics = load_all_metrics(args.root)
    metrics_by_method = _group_metrics_by_method(all_metrics)

    available_scenarios = {scenario for _, scenario in all_metrics.keys()}
    scenario_params = _scenario_parameters(scenarios_cfg, available_scenarios)
    if not scenario_params:
        raise RuntimeError("Aucun scénario muni de N et period n'a été trouvé dans la configuration.")

    n_values = sorted({params["N"] for params in scenario_params.values()})
    period_values = sorted({params["period"] for params in scenario_params.values()})

    for method, metrics_by_scenario in metrics_by_method.items():
        method_tag = _sanitize_method_name(method)
        pdr_matrix = _build_matrix(
            metrics_by_scenario,
            scenarios_cfg,
            scenario_params,
            n_values,
            period_values,
            metric="pdr_global",
        )
        der_matrix = _build_matrix(
            metrics_by_scenario,
            scenarios_cfg,
            scenario_params,
            n_values,
            period_values,
            metric="der_global",
        )
        gap_matrix = _build_matrix(
            metrics_by_scenario,
            scenarios_cfg,
            scenario_params,
            n_values,
            period_values,
            metric="target_gap",
        )

        _plot_heatmap(
            pdr_matrix,
            n_values,
            period_values,
            title=f"Global PDR – {method}",
            colorbar_label="PDR",
            output_path=args.out / f"pdr_heatmap_{method_tag}.png",
            cmap="viridis",
        )
        _plot_heatmap(
            der_matrix,
            n_values,
            period_values,
            title=f"DER – {method}",
            colorbar_label="DER",
            output_path=args.out / f"der_heatmap_{method_tag}.png",
            cmap="magma",
        )
        _plot_heatmap(
            gap_matrix,
            n_values,
            period_values,
            title=f"Target gap – {method}",
            colorbar_label="PDR - target",
            output_path=args.out / f"target_gap_heatmap_{method_tag}.png",
            cmap="coolwarm",
        )


if __name__ == "__main__":  # pragma: no cover - exécution CLI
    main()
