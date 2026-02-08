"""Génération des figures pour le mini banc QoS clusters."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

from article_c.common.plotting_style import apply_base_rcparams
from article_c.common.plot_helpers import apply_figure_layout
from metrics import RunMetrics, load_cluster_ids
from plot_defaults import resolve_ieee_figsize


DEFAULT_ALGORITHMS = [
    "ADR",
    "APRA-like",
    "MixRA-H",
    "MixRA-Opt",
]

PALETTE = [
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
THRESHOLDS_DB = [(-6.0, "Seuil décodage"), (6.0, "Capture")]
SMALL_VALUE_UPPER = 0.3


def _add_thresholds(ax):
    for value, label in THRESHOLDS_DB:
        ax.axvline(value, color="grey", linestyle="--", linewidth=1.0)
        ax.text(value, ax.get_ylim()[1], f" {label}", rotation=90, va="bottom", ha="left", fontsize=8)


def _apply_small_value_ylim(ax, values: Sequence[float], upper: float = SMALL_VALUE_UPPER) -> None:
    cleaned = [value for value in values if value == value]
    if cleaned and max(cleaned) <= upper:
        ax.set_ylim(0.0, upper)


def _log_floor(values: Sequence[float]) -> float | None:
    positives = [value for value in values if value == value and value > 0]
    if not positives:
        return None
    return min(positives) / 10


def _sanitize_log_values(values: Sequence[float], floor: float) -> List[float]:
    sanitized: List[float] = []
    for value in values:
        if value != value:
            sanitized.append(float("nan"))
        elif value <= 0:
            sanitized.append(floor)
        else:
            sanitized.append(value)
    return sanitized


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


def _style_mapping(labels: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for index, label in enumerate(labels):
        mapping[label] = PALETTE[index % len(PALETTE)]
    return mapping


def _scenario_metadata(results: Sequence[RunMetrics]) -> Dict[str, Tuple[int, float]]:
    mapping: Dict[str, Tuple[int, float]] = {}
    for item in results:
        if item.scenario not in mapping:
            mapping[item.scenario] = (item.num_nodes, item.period_s)
    return mapping


def _subtitle_for_scenarios(metadata: Mapping[str, Tuple[int, float]], scenarios: Sequence[str]) -> str:
    parts: List[str] = []
    for scenario in scenarios:
        if scenario not in metadata:
            continue
        nodes, period = metadata[scenario]
        parts.append(f"{scenario}: {nodes} nœuds, période {period:.0f}s")
    return " | ".join(parts)


def _plot_pdr_by_cluster(
    results: Sequence[RunMetrics],
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    cluster_ids = load_cluster_ids(results)
    if not cluster_ids:
        return None
    base_width, base_height = resolve_ieee_figsize(len(algorithms))
    fig, axes = plt.subplots(
        1,
        len(cluster_ids),
        figsize=(base_width, base_height),
        sharey=True,
    )
    if len(cluster_ids) == 1:
        axes = [axes]  # type: ignore[list-item]
    hline_added = False
    styles = _style_mapping(algorithms)
    for axis, cluster_id in zip(axes, cluster_ids):
        for algorithm in algorithms:
            values: List[float] = []
            for scenario in scenarios:
                metrics = mapping.get((scenario, algorithm))
                values.append(metrics.cluster_pdr.get(cluster_id, float("nan")) if metrics else float("nan"))
            if any(value == value for value in values):
                axis.plot(
                    scenarios,
                    values,
                    marker="o",
                    label=algorithm,
                    color=styles.get(algorithm),
                )
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
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(handles))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.08, 1.0, 0.88)})
    output_path = output_dir / "pdr_clusters.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_pdr_global(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    plotted = False
    styles = _style_mapping(algorithms)
    bar_width = 0.8 / max(len(algorithms), 1)
    x_positions = list(range(len(scenarios)))
    for index, algorithm in enumerate(algorithms):
        heights: List[float] = []
        errors: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            heights.append(metrics.pdr_global if metrics else float("nan"))
            errors.append(metrics.pdr_ci95 if metrics else 0.0)
        offset = (index - (len(algorithms) - 1) / 2) * bar_width
        positions = [x + offset for x in x_positions]
        if any(value == value for value in heights):
            ax.bar(
                positions,
                heights,
                width=bar_width,
                yerr=errors,
                color=styles.get(algorithm),
                label=algorithm,
                capsize=4,
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("PDR global")
    ax.set_xlabel("Scénario")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "pdr_global.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_der_global(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    plotted = False
    styles = _style_mapping(algorithms)
    bar_width = 0.8 / max(len(algorithms), 1)
    x_positions = list(range(len(scenarios)))
    all_heights: List[float] = []
    for index, algorithm in enumerate(algorithms):
        heights: List[float] = []
        errors: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            heights.append(metrics.der_global if metrics else float("nan"))
            errors.append(metrics.der_ci95 if metrics else 0.0)
        all_heights.extend(heights)
        offset = (index - (len(algorithms) - 1) / 2) * bar_width
        positions = [x + offset for x in x_positions]
        if any(value == value for value in heights):
            ax.bar(
                positions,
                heights,
                width=bar_width,
                yerr=errors,
                color=styles.get(algorithm),
                label=algorithm,
                capsize=4,
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("DER global")
    ax.set_xlabel("Scénario")
    ax.set_title("DER global (échelle linéaire)")
    ax.set_ylim(0.0, 1.05)
    _apply_small_value_ylim(ax, all_heights)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "der_global.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_der_global_log(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    plotted = False
    styles = _style_mapping(algorithms)
    bar_width = 0.8 / max(len(algorithms), 1)
    x_positions = list(range(len(scenarios)))
    all_heights: List[float] = []
    all_errors: List[float] = []
    heights_by_algo: List[List[float]] = []
    errors_by_algo: List[List[float]] = []
    for algorithm in algorithms:
        heights: List[float] = []
        errors: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            heights.append(metrics.der_global if metrics else float("nan"))
            errors.append(metrics.der_ci95 if metrics else 0.0)
        heights_by_algo.append(heights)
        errors_by_algo.append(errors)
        all_heights.extend(heights)
        all_errors.extend(errors)
    floor = _log_floor(all_heights)
    if floor is None:
        plt.close(fig)
        return None
    for index, algorithm in enumerate(algorithms):
        heights = _sanitize_log_values(heights_by_algo[index], floor)
        errors = errors_by_algo[index]
        offset = (index - (len(algorithms) - 1) / 2) * bar_width
        positions = [x + offset for x in x_positions]
        if any(value == value for value in heights):
            ax.bar(
                positions,
                heights,
                width=bar_width,
                yerr=errors,
                color=styles.get(algorithm),
                label=algorithm,
                capsize=4,
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("DER global")
    ax.set_xlabel("Scénario")
    ax.set_title("DER global (échelle log)")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "der_global_log.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_snir_cdf(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenario: str,
    algorithms: Sequence[str],
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
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
    apply_figure_layout(fig, tight_layout=True)
    output_path = output_dir / "snir_cdf.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_snir_distributions(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    if not scenarios:
        return None
    base_width, base_height = resolve_ieee_figsize(len(algorithms))
    fig, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(base_width, base_height * len(scenarios)),
        sharex="col",
        squeeze=False,
    )
    styles = _style_mapping(algorithms)
    plotted = False
    for row, scenario in enumerate(scenarios):
        hist_ax = axes[row][0]
        cdf_ax = axes[row][1]
        for algorithm in algorithms:
            metrics = mapping.get((scenario, algorithm))
            if not metrics or not metrics.snir_values:
                continue
            hist_ax.hist(
                metrics.snir_values,
                bins=20,
                density=True,
                alpha=0.35,
                color=styles.get(algorithm),
                label=algorithm if row == 0 else None,
            )
            xs = [value for value, _ in metrics.snir_cdf]
            ys = [prob for _, prob in metrics.snir_cdf]
            cdf_ax.step(xs, ys, where="post", label=algorithm if row == 0 else None, color=styles.get(algorithm))
            plotted = True
        hist_ax.set_ylabel("Densité")
        hist_ax.set_title(f"{scenario} – Histogramme SNIR")
        hist_ax.grid(True, linestyle=":", alpha=0.5)
        _add_thresholds(hist_ax)
        cdf_ax.set_ylabel("CDF")
        cdf_ax.set_title(f"{scenario} – CDF SNIR")
        cdf_ax.grid(True, linestyle=":", alpha=0.5)
        _add_thresholds(cdf_ax)
    for col in range(2):
        axes[-1][col].set_xlabel("SNIR (dB)")
    if not plotted:
        plt.close(fig)
        return None
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "snir_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _compute_binned_rates(entries: Sequence[Tuple[float, str]], bin_edges: Sequence[float]):
    counts = [0 for _ in range(len(bin_edges) - 1)]
    successes = [0 for _ in range(len(bin_edges) - 1)]
    for value, result in entries:
        if value != value:
            continue
        for idx in range(len(bin_edges) - 1):
            if bin_edges[idx] <= value < bin_edges[idx + 1]:
                counts[idx] += 1
                if str(result).lower().startswith("success"):
                    successes[idx] += 1
                break
    rates = []
    for count, ok in zip(counts, successes):
        rates.append(ok / count if count else float("nan"))
    centers = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    return centers, rates


def _plot_rates_vs_snir(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    base_width, base_height = resolve_ieee_figsize(len(algorithms))
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(base_width, base_height * len(scenarios)),
        sharex=True,
    )
    if len(scenarios) == 1:
        axes = [axes]  # type: ignore[list-item]
    styles = _style_mapping(algorithms)
    plotted = False
    for row, scenario in enumerate(scenarios):
        axis = axes[row]
        min_val = math.inf
        max_val = -math.inf
        for algorithm in algorithms:
            metrics = mapping.get((scenario, algorithm))
            if metrics and metrics.snir_values:
                min_val = min(min_val, min(metrics.snir_values))
                max_val = max(max_val, max(metrics.snir_values))
        if not math.isfinite(min_val) or not math.isfinite(max_val):
            continue
        bin_edges = list(
            v for v in [min_val + step * (max_val - min_val) / 12 for step in range(13)]
        )
        for algorithm in algorithms:
            metrics = mapping.get((scenario, algorithm))
            if not metrics or not metrics.snir_by_result:
                continue
            centers, rates = _compute_binned_rates(metrics.snir_by_result, bin_edges)
            axis.plot(centers, rates, marker="o", color=styles.get(algorithm), label=algorithm)
            plotted = True
        axis.set_ylabel("PDR par bin SNIR")
        axis.set_title(f"{scenario} – Impact du canal")
        axis.grid(True, linestyle=":", alpha=0.5)
        _add_thresholds(axis)
    if not plotted:
        plt.close(fig)
        return None
    axes[-1].set_xlabel("SNIR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "pdr_vs_snir.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_collisions_vs_load(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    styles = _style_mapping(algorithms)
    plotted = False
    all_rates: List[float] = []
    for algorithm in algorithms:
        xs: List[float] = []
        ys: List[float] = []
        sizes: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            if not metrics or metrics.attempted == 0:
                continue
            load = metrics.num_nodes / metrics.period_s if metrics.period_s > 0 else float("nan")
            collision_rate = metrics.failures_collision / metrics.attempted
            xs.append(load)
            ys.append(collision_rate)
            sizes.append(max(metrics.attempted / 50, 10))
        if xs:
            ax.scatter(xs, ys, s=sizes, alpha=0.7, color=styles.get(algorithm), label=algorithm)
            plotted = True
        all_rates.extend(ys)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Charge (nœuds / période en s)")
    ax.set_ylabel("Taux de collisions")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Collisions en fonction de la charge (échelle linéaire)")
    _apply_small_value_ylim(ax, all_rates)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "collisions_vs_charge.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_collisions_vs_load_log(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    styles = _style_mapping(algorithms)
    plotted = False
    all_rates: List[float] = []
    data_by_algo: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for algorithm in algorithms:
        xs: List[float] = []
        ys: List[float] = []
        sizes: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            if not metrics or metrics.attempted == 0:
                continue
            load = metrics.num_nodes / metrics.period_s if metrics.period_s > 0 else float("nan")
            collision_rate = metrics.failures_collision / metrics.attempted
            xs.append(load)
            ys.append(collision_rate)
            sizes.append(max(metrics.attempted / 50, 10))
        if xs:
            data_by_algo[algorithm] = (xs, ys, sizes)
            all_rates.extend(ys)
    floor = _log_floor(all_rates)
    if floor is None:
        plt.close(fig)
        return None
    for algorithm, (xs, ys, sizes) in data_by_algo.items():
        ys_log = _sanitize_log_values(ys, floor)
        ax.scatter(xs, ys_log, s=sizes, alpha=0.7, color=styles.get(algorithm), label=algorithm)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Charge (nœuds / période en s)")
    ax.set_ylabel("Taux de collisions")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Collisions en fonction de la charge (échelle log)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(algorithms))
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "collisions_vs_charge_log.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_delivery_breakdown(
    mapping: Mapping[Tuple[str, str], RunMetrics],
    output_dir: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    subtitle: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    styles = _style_mapping(algorithms)
    x_positions = list(range(len(scenarios)))
    bar_width = 0.8 / max(len(algorithms), 1)
    plotted = False
    for algo_index, algorithm in enumerate(algorithms):
        success_values: List[float] = []
        collision_values: List[float] = []
        nosig_values: List[float] = []
        for scenario in scenarios:
            metrics = mapping.get((scenario, algorithm))
            if metrics and metrics.attempted > 0:
                success = metrics.delivered / metrics.attempted
                collision = metrics.failures_collision / metrics.attempted
                nosig = metrics.failures_no_signal / metrics.attempted
            else:
                success = collision = nosig = float("nan")
            success_values.append(success)
            collision_values.append(collision)
            nosig_values.append(nosig)
        offset = (algo_index - (len(algorithms) - 1) / 2) * bar_width
        positions = [x + offset for x in x_positions]
        if any(value == value for value in success_values):
            ax.bar(positions, success_values, width=bar_width, color=styles.get(algorithm), label=f"{algorithm} – succès")
            ax.bar(
                positions,
                collision_values,
                width=bar_width,
                bottom=success_values,
                color="#f4a261",
                label=f"{algorithm} – collisions",
            )
            bottom = [s + c if (s == s and c == c) else float("nan") for s, c in zip(success_values, collision_values)]
            ax.bar(
                positions,
                nosig_values,
                width=bar_width,
                bottom=bottom,
                color="#e76f51",
                label=f"{algorithm} – sans signal",
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Répartition des tentatives")
    ax.set_xlabel("Scénario")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    if subtitle:
        fig.suptitle(subtitle)
    apply_figure_layout(fig, tight_layout={"rect": (0.0, 0.05, 1.0, 0.88)})
    output_path = output_dir / "breakdown_tentatives.png"
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
    """Génère l'ensemble des figures demandées et retourne leur chemin."""

    if not results:
        return []
    apply_base_rcparams()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mapping = _index_results(results)
    scenarios = _resolve_order((result.scenario for result in results), scenario_order)
    algorithms = _resolve_order((result.algorithm for result in results), algorithm_order or DEFAULT_ALGORITHMS)
    metadata = _scenario_metadata(results)
    subtitle = _subtitle_for_scenarios(metadata, scenarios)

    generated: List[Path] = []
    pdr_path = _plot_pdr_by_cluster(results, mapping, output_path, scenarios, algorithms, subtitle)
    if pdr_path is not None:
        generated.append(pdr_path)
    pdr_global_path = _plot_pdr_global(mapping, output_path, scenarios, algorithms, subtitle)
    if pdr_global_path is not None:
        generated.append(pdr_global_path)
    der_path = _plot_der_global(mapping, output_path, scenarios, algorithms, subtitle)
    if der_path is not None:
        generated.append(der_path)
    der_log_path = _plot_der_global_log(mapping, output_path, scenarios, algorithms, subtitle)
    if der_log_path is not None:
        generated.append(der_log_path)
    breakdown_path = _plot_delivery_breakdown(mapping, output_path, scenarios, algorithms, subtitle)
    if breakdown_path is not None:
        generated.append(breakdown_path)
    chosen_scenario = cdf_scenario or (scenarios[1] if len(scenarios) > 1 else scenarios[0])
    snir_path = _plot_snir_cdf(mapping, output_path, chosen_scenario, algorithms)
    if snir_path is not None:
        generated.append(snir_path)
    snir_dist_path = _plot_snir_distributions(mapping, output_path, scenarios, algorithms, subtitle)
    if snir_dist_path is not None:
        generated.append(snir_dist_path)
    snir_rate_path = _plot_rates_vs_snir(mapping, output_path, scenarios, algorithms, subtitle)
    if snir_rate_path is not None:
        generated.append(snir_rate_path)
    collision_path = _plot_collisions_vs_load(mapping, output_path, scenarios, algorithms, subtitle)
    if collision_path is not None:
        generated.append(collision_path)
    collision_log_path = _plot_collisions_vs_load_log(mapping, output_path, scenarios, algorithms, subtitle)
    if collision_log_path is not None:
        generated.append(collision_log_path)
    return generated


__all__ = ["generate_plots"]
