"""Génération des figures pour le mini banc QoS clusters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

from metrics import RunMetrics, load_cluster_ids


DEFAULT_ALGORITHMS = [
    "ADR",
    "APRA-like",
    "MixRA-H",
    "MixRA-Opt",
]


def _index_results(results: Sequence[RunMetrics]) -> Dict[Tuple[str, str], RunMetrics]:
    return {(item.scenario, item.algorithm): item for item in results}


def _resolve_order(values: Iterable[str], preferred: Sequence[str] | None) -> List[str]:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if preferred:
        ordered = [value for value in preferred if value in unique]
        for value in unique:
            if value not in ordered:
                ordered.append(value)
        return ordered
    return unique


def _plot_pdr_by_cluster(
    results: Sequence[RunMetrics],
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
) -> Path | None:
    cluster_ids = load_cluster_ids(results)
    if not cluster_ids:
        return None
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(5.0 * len(cluster_ids), 4.0), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]  # type: ignore[list-item]
    hline_added = False
    for axis, cluster_id in zip(axes, cluster_ids):
        for algorithm in algorithms:
            values: List[float] = []
            for scenario in scenarios:
                metrics = mapping.get((scenario, algorithm))
                values.append(metrics.cluster_pdr.get(cluster_id, float("nan")) if metrics else float("nan"))
            if any(value == value for value in values):
                axis.plot(scenarios, values, marker="o", label=algorithm)
        target = None
        for scenario in scenarios:
            for algorithm in algorithms:
                metrics = mapping.get((scenario, algorithm))
                if metrics and cluster_id in metrics.cluster_targets:
                    target = metrics.cluster_targets[cluster_id]
                    break
            if target is not None:
                break
        if target is not None:
            axis.axhline(target, color="grey", linestyle="--", linewidth=1.0, label="Cible" if not hline_added else None)
            hline_added = True
        axis.set_title(f"Cluster {cluster_id}")
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Scénario")
        axis.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("PDR")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    output_path = output_dir / "pdr_clusters.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_der_global(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = False
    for algorithm in algorithms:
        values: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            values.append(metrics.der_global if metrics else float("nan"))
        if any(value == value for value in values):
            ax.plot(scenarios, values, marker="o", label=algorithm)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_ylabel("DER global")
    ax.set_xlabel("Scénario")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / "der_global.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_snir_cdf(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenario: str,
    algorithms: Sequence[str],
) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = False
    for algorithm in algorithms:
        metrics = mapping.get((scenario, algorithm))
        if not metrics or not metrics.snir_cdf:
            continue
        xs = [value for value, _ in metrics.snir_cdf]
        ys = [prob for _, prob in metrics.snir_cdf]
        ax.step(xs, ys, where="post", label=algorithm)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("SNIR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF SNIR – {scenario}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = output_dir / "snir_cdf.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_plots(
    results: Sequence[RunMetrics],
    output_dir: str | Path,
    *,
    scenario_order: Sequence[str] | None = None,
    algorithm_order: Sequence[str] | None = None,
    cdf_scenario: str | None = None,
) -> List[Path]:
    """Génère les trois figures demandées et retourne leur chemin."""

    if not results:
        return []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mapping = _index_results(results)
    scenarios = _resolve_order((result.scenario for result in results), scenario_order)
    algorithms = _resolve_order((result.algorithm for result in results), algorithm_order or DEFAULT_ALGORITHMS)

    generated: List[Path] = []
    pdr_path = _plot_pdr_by_cluster(results, mapping, output_path, scenarios, algorithms)
    if pdr_path is not None:
        generated.append(pdr_path)
    der_path = _plot_der_global(mapping, output_path, scenarios, algorithms)
    if der_path is not None:
        generated.append(der_path)
    chosen_scenario = cdf_scenario or (scenarios[1] if len(scenarios) > 1 else scenarios[0])
    snir_path = _plot_snir_cdf(mapping, output_path, chosen_scenario, algorithms)
    if snir_path is not None:
        generated.append(snir_path)
    return generated


__all__ = ["generate_plots"]
