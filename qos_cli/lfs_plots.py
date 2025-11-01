"""CLI pour générer des visualisations QoS à partir des résultats LoRaFlexSim."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from .lfs_metrics import (
    MethodScenarioMetrics,
    load_all_metrics,
    load_yaml_config,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit l'espace de noms d'arguments pour la CLI de génération de graphiques."""

    parser = argparse.ArgumentParser(
        prog="lfs_plots",
        description="Génère des figures (PDR, DER, CDF SNIR) à partir des résultats QoS.",
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

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        clusters = sorted(
            {
                cluster
                for metric in method_metrics.values()
                for cluster in metric.cluster_pdr.keys()
            }
        )
        if not clusters:
            ax.text(0.5, 0.5, "Aucune donnée PDR par cluster", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.0)
            continue
        for cluster in clusters:
            values: List[float] = []
            for scenario in scenarios:
                metric = method_metrics.get(scenario)
                values.append(metric.cluster_pdr.get(cluster, float("nan")) if metric else float("nan"))
            # TODO: améliorer la palette de couleurs et le style de ligne pour distinguer les clusters.
            ax.plot(x_positions, values, marker="o", label=str(cluster))
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Méthode : {method}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize="small")

    axes[-1].set_xticks(x_positions, scenarios)
    axes[-1].set_xlabel("Scénario")
    fig.suptitle("PDR par cluster en fonction des scénarios")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = out_dir / "pdr_clusters_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_der(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    """Trace la DER globale par scénario pour chaque méthode."""

    if not scenarios or not metrics_by_method:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
    x_positions = list(range(len(scenarios)))

    for method in sorted(metrics_by_method.keys()):
        values: List[float] = []
        for scenario in scenarios:
            metric = metrics_by_method[method].get(scenario) if method in metrics_by_method else None
            value = metric.der_global if metric and metric.der_global is not None else float("nan")
            values.append(value)
        # TODO: harmoniser les marqueurs et couleurs pour améliorer la lisibilité.
        ax.plot(x_positions, values, marker="o", label=method)

    ax.set_xticks(x_positions, scenarios)
    ax.set_xlabel("Scénario")
    ax.set_ylabel("DER globale")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("DER globale par scénario")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = out_dir / "der_global_vs_scenarios.png"
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
    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        has_data = False
        for method in sorted(metrics_by_method.keys()):
            metric = metrics_by_method[method].get(scenario)
            if not metric or not metric.snir_cdf:
                continue
            has_data = True
            xs, ys = zip(*sorted(metric.snir_cdf))
            # TODO: ajouter des annotations (p. ex. médiane SNIR) pour enrichir la lecture.
            ax.step(xs, ys, where="post", label=method)
        if not has_data:
            ax.text(0.5, 0.5, "Données SNIR indisponibles", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("SNIR (dB)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(auto=True)
        ax.set_title(f"CDF SNIR – scénario {scenario}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        fig.tight_layout()

        filename = f"snir_cdf_{sanitize_filename(scenario)}.png"
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)
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

    all_metrics = load_all_metrics(metrics_root)
    if not all_metrics:
        raise RuntimeError("Aucune métrique détectée – vérifiez la structure des résultats.")

    metrics_by_method = build_method_mapping(all_metrics)
    scenarios = ordered_scenarios(all_metrics, scenarios_cfg)
    if not scenarios:
        scenarios = sorted({scenario for _, scenario in all_metrics.keys()})

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_pdr(metrics_by_method, scenarios, out_dir)
    plot_der(metrics_by_method, scenarios, out_dir)
    plot_snir_cdf(metrics_by_method, scenarios, out_dir)


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    main()
