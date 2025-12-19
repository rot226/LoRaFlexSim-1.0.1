"""CLI pour générer des visualisations QoS à partir des résultats LoRaFlexSim."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

try:  # pragma: no cover - dépend du mode d'exécution
    from .lfs_metrics import (
        MethodScenarioMetrics,
        load_all_metrics,
        load_yaml_config,
    )
except ImportError:  # pragma: no cover - fallback pour exécution directe
    from lfs_metrics import (
        MethodScenarioMetrics,
        load_all_metrics,
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

SNIR_STATE_COLORS = {
    "snir_on": "#d62728",  # rouge
    "snir_off": "#1f77b4",  # bleu
    "snir_unknown": "#7f7f7f",
}

SNIR_STATE_LABELS = {
    "snir_on": "SNIR activé",
    "snir_off": "SNIR désactivé",
    "snir_unknown": "SNIR inconnu",
}


def _style_mapping(labels: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        mapping[str(label)] = (color, marker)
    return mapping


def _values_for_attribute(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    attribute: str,
) -> Dict[str, List[float]]:
    values_per_method: Dict[str, List[float]] = {}
    for method, scenario_metrics in metrics_by_method.items():
        values: List[float] = []
        for scenario in scenarios:
            metric = scenario_metrics.get(scenario)
            value = getattr(metric, attribute) if metric else None
            if value is None:
                values.append(float("nan"))
            else:
                values.append(float(value))
        values_per_method[method] = values
    return values_per_method


def _metric_snir_state(metric: Optional[MethodScenarioMetrics]) -> str:
    if metric is None:
        return "snir_unknown"
    if metric.use_snir is True:
        return "snir_on"
    if metric.use_snir is False:
        return "snir_off"
    if metric.snir_state:
        normalized = metric.snir_state.strip().lower()
        if normalized in SNIR_STATE_LABELS:
            return normalized
        if normalized in {"snir-on", "on", "enabled"}:
            return "snir_on"
        if normalized in {"snir-off", "off", "disabled"}:
            return "snir_off"
    return "snir_unknown"


def _values_by_snir_state(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    attribute: str,
) -> Dict[str, Dict[str, List[float]]]:
    states = {"snir_on", "snir_off", "snir_unknown"}
    values_per_state: Dict[str, Dict[str, List[float]]] = {
        state: {method: [float("nan")] * len(scenarios) for method in metrics_by_method}
        for state in states
    }

    for method, scenario_metrics in metrics_by_method.items():
        for idx, scenario in enumerate(scenarios):
            metric = scenario_metrics.get(scenario)
            if metric is None:
                continue
            value = getattr(metric, attribute, None)
            if value is None:
                continue
            state = _metric_snir_state(metric)
            if state not in values_per_state:
                continue
            values_per_state[state][method][idx] = float(value)
    return values_per_state


def _all_nan(values: Sequence[float]) -> bool:
    return all(math.isnan(value) for value in values)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit l'espace de noms d'arguments pour la CLI de génération de graphiques."""

    parser = argparse.ArgumentParser(
        prog="lfs_plots",
        description=(
            "Génère des figures (PDR global/cluster, DER, collisions, énergie, équité, part SF et CDF SNIR) à partir des résultats QoS."
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
        required=False,
        help="Fichier YAML décrivant les scénarios (utilisé pour ordonner les sorties).",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=Path("qos_cli") / "figures",
        help="Dossier de destination des figures (défaut : qos_cli/figures).",
    )
    return parser.parse_args(argv)


def build_method_mapping(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics]
) -> Dict[str, Dict[str, MethodScenarioMetrics]]:
    """Réorganise les métriques par méthode puis par scénario."""

    grouped: Dict[str, Dict[str, MethodScenarioMetrics]] = {}
    for (method, scenario), data in metrics.items():
        grouped.setdefault(method, {})[scenario] = data
    return grouped


def ordered_scenarios(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
    config: Optional[Mapping[str, Mapping[str, object]]],
) -> List[str]:
    """Retourne la liste ordonnée des scénarios détectés."""

    if config:
        order = [str(key) for key in config.keys()]
    else:
        order = sorted({scenario for _, scenario in metrics.keys()})
    available = {scenario for _, scenario in metrics.keys()}
    return [scenario for scenario in order if scenario in available]


def plot_cluster_pdr(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    """Trace le PDR par cluster pour chaque méthode en fonction du scénario."""

    if not scenarios or not metrics_by_method:
        return None

    methods = sorted(metrics_by_method.keys())
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), max(3.5, 2.5 * len(methods))),
    )
    if len(methods) == 1:
        axes = [axes]  # type: ignore[assignment]

    x_positions = list(range(len(scenarios)))

    method_styles = _style_mapping(methods)

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        clusters = sorted(
            {
                cluster
                for metric in method_metrics.values()
                for cluster in metric.cluster_pdr.keys()
            }
        )
        cluster_styles = _style_mapping(clusters)
        if not clusters:
            ax.text(
                0.5,
                0.5,
                "No cluster PDR data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.0)
            continue
        for cluster in clusters:
            values: List[float] = []
            for scenario in scenarios:
                metric = method_metrics.get(scenario)
                values.append(metric.cluster_pdr.get(cluster, float("nan")) if metric else float("nan"))
            if _all_nan(values):
                continue
            color, marker = cluster_styles[str(cluster)]
            ax.plot(
                x_positions,
                values,
                marker=marker,
                color=color,
                label=str(cluster),
            )
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.text(
                0.5,
                0.5,
                "Clusters unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    for ax, method in zip(axes, methods):
        color, _ = method_styles[method]
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.2)

    axes[-1].set_xticks(x_positions, scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "pdr_clusters_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metric_with_snir_states(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    attribute: str,
    ylabel: str,
    filename_base: str,
) -> List[Path]:
    if not scenarios or not metrics_by_method:
        return []

    x_positions = list(range(len(scenarios)))
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    values_by_state = _values_by_snir_state(metrics_by_method, scenarios, attribute)

    saved_paths: List[Path] = []

    def render(states_to_plot: List[str], suffix: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
        plotted = False
        for state in states_to_plot:
            state_values = values_by_state.get(state, {})
            color = SNIR_STATE_COLORS.get(state, "#7f7f7f")
            label_state = SNIR_STATE_LABELS.get(state, state)
            for method, values in state_values.items():
                if not values or _all_nan(values):
                    continue
                _, marker = method_styles[method]
                ax.plot(
                    x_positions,
                    values,
                    marker=marker,
                    color=color,
                    label=f"{method} ({label_state})",
                )
                plotted = True

        ax.set_xticks(x_positions, scenarios)
        ax.set_xlabel("Scenario")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if title:
            ax.set_title(title)
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                f"{ylabel} unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()

        filename = f"{filename_base}{suffix}.png"
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    render(["snir_on"], "_snir-on", f"{ylabel} – SNIR activé")
    render(["snir_off"], "_snir-off", f"{ylabel} – SNIR désactivé")
    render(["snir_on", "snir_off", "snir_unknown"], "_snir-mixed", f"{ylabel} – SNIR superposé")

    return saved_paths


def plot_der(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    """Trace la DER globale par scénario en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="der_global",
        ylabel="Global DER",
        filename_base="der_global_vs_scenarios",
    )


def plot_pdr(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    """Trace le PDR global en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="pdr_global",
        ylabel="Global PDR",
        filename_base="pdr_global_vs_scenarios",
    )


def plot_collisions(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
    x_positions = list(range(len(scenarios)))
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    plotted = False

    collision_values = _values_for_attribute(metrics_by_method, scenarios, "collisions")
    for method in sorted(collision_values.keys()):
        values = collision_values[method]
        if not values or _all_nan(values):
            continue
        color, marker = method_styles[method]
        ax.plot(x_positions, values, marker=marker, color=color, label=method)
        plotted = True

    ax.set_xticks(x_positions, scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Uplink collisions")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0.0)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Collision data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()

    output_path = out_dir / "collisions_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_energy(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
    x_positions = list(range(len(scenarios)))
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    plotted = False

    energy_values = _values_for_attribute(metrics_by_method, scenarios, "energy_j")
    for method in sorted(energy_values.keys()):
        values = energy_values[method]
        if not values or _all_nan(values):
            continue
        color, marker = method_styles[method]
        ax.plot(x_positions, values, marker=marker, color=color, label=method)
        plotted = True

    ax.set_xticks(x_positions, scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total energy (J)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0.0)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Energy data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()

    output_path = out_dir / "energy_total_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_jain_index(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
    x_positions = list(range(len(scenarios)))
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    plotted = False

    jain_values = _values_for_attribute(metrics_by_method, scenarios, "jain_index")
    for method in sorted(jain_values.keys()):
        values = jain_values[method]
        if not values or _all_nan(values):
            continue
        color, marker = method_styles[method]
        ax.plot(x_positions, values, marker=marker, color=color, label=method)
        plotted = True

    ax.set_xticks(x_positions, scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Jain index")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Jain index unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()

    output_path = out_dir / "jain_index_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_min_sf_share(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
    x_positions = list(range(len(scenarios)))
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    plotted = False

    min_sf_values = _values_for_attribute(metrics_by_method, scenarios, "min_sf_share")
    for method in sorted(min_sf_values.keys()):
        values = min_sf_values[method]
        if not values or _all_nan(values):
            continue
        color, marker = method_styles[method]
        ax.plot(x_positions, values, marker=marker, color=color, label=method)
        plotted = True

    ax.set_xticks(x_positions, scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Minimum SF share")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "SF distribution unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()

    output_path = out_dir / "min_sf_share_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def sanitize_filename(text: str) -> str:
    """Transforme un scénario en identifiant de fichier valide."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return cleaned.strip("_") or "scenario"


def plot_snir_cdf(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    """Génère les CDF SNIR par scénario avec une courbe par méthode."""

    saved_paths: List[Path] = []
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))
    states_order = ["snir_on", "snir_off", "snir_unknown"]

    for scenario in scenarios:
        def render(state_filter: List[str], suffix: str, title: str) -> None:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            has_data = False
            for method in sorted(metrics_by_method.keys()):
                metric = metrics_by_method[method].get(scenario)
                if not metric or not metric.snir_cdf:
                    continue
                state = _metric_snir_state(metric)
                if state_filter and state not in state_filter:
                    continue
                has_data = True
                xs, ys = zip(*sorted(metric.snir_cdf))
                _, marker = method_styles[method]
                color = SNIR_STATE_COLORS.get(state, "#7f7f7f")
                label_state = SNIR_STATE_LABELS.get(state, state)
                ax.step(
                    xs,
                    ys,
                    where="post",
                    label=f"{method} ({label_state})",
                    color=color,
                )
                ax.plot([], [], marker=marker, color=color, linestyle="", label="")
            if not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "SNIR data unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_xlabel("SNIR (dB)")
            ax.set_ylabel("CDF")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(auto=True)
            ax.grid(True, linestyle="--", alpha=0.4)
            if title:
                ax.set_title(title)
            handles, labels = ax.get_legend_handles_labels()
            filtered_handles = [h for h, l in zip(handles, labels) if l]
            filtered_labels = [l for l in labels if l]
            if filtered_handles:
                ax.legend(filtered_handles, filtered_labels, loc="best")
            fig.tight_layout()

            filename = f"snir_cdf_{sanitize_filename(scenario)}{suffix}.png"
            output_path = out_dir / filename
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            saved_paths.append(output_path)

        render(["snir_on"], "_snir-on", f"SNIR CDF – {scenario} (SNIR activé)")
        render(["snir_off"], "_snir-off", f"SNIR CDF – {scenario} (SNIR désactivé)")
        render(states_order, "_snir-mixed", f"SNIR CDF – {scenario} (superposé)")
    return saved_paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics_root = args.root
    config_path: Optional[Path] = args.config
    out_dir: Path = args.out

    if not metrics_root.exists():
        raise FileNotFoundError(f"Dossier de résultats introuvable : {metrics_root}")

    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]] = None
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")
        scenarios_cfg = load_yaml_config(config_path)

    all_metrics = load_all_metrics(metrics_root, scenarios_cfg)
    if not all_metrics:
        raise RuntimeError("Aucune métrique détectée – vérifiez la structure des résultats.")

    metrics_by_method = build_method_mapping(all_metrics)
    scenarios = ordered_scenarios(all_metrics, scenarios_cfg)
    if not scenarios:
        scenarios = sorted({scenario for _, scenario in all_metrics.keys()})

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_pdr(metrics_by_method, scenarios, out_dir)
    plot_pdr(metrics_by_method, scenarios, out_dir)
    plot_der(metrics_by_method, scenarios, out_dir)
    plot_collisions(metrics_by_method, scenarios, out_dir)
    plot_energy(metrics_by_method, scenarios, out_dir)
    plot_jain_index(metrics_by_method, scenarios, out_dir)
    plot_min_sf_share(metrics_by_method, scenarios, out_dir)
    plot_snir_cdf(metrics_by_method, scenarios, out_dir)


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    main()
