"""Trace les figures de comparaison Step 2 à partir des CSV normalisés."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_step1_results import SNIR_COLORS

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step2"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step2"

STATE_LABELS = ["snir_off", "snir_on"]
SNIR_TITLES = {"snir_off": "SNIR OFF", "snir_on": "SNIR ON"}

MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]
DEFAULT_ALGO_PRIORITY = ["adr", "apra", "mixra_h", "mixra_opt"]


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean = fmean(values)
    std = pstdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, margin


def _valid_values(values: Iterable[float | None]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if value is None or math.isnan(value):
            continue
        cleaned.append(float(value))
    return cleaned


def _select_algorithms(records: Iterable[Mapping[str, Any]], selected: Sequence[str] | None) -> List[str]:
    if selected:
        return list(selected)
    available = {str(record["algorithm"]) for record in records if record.get("algorithm")}
    algorithms = [algo for algo in DEFAULT_ALGO_PRIORITY if algo in available]
    if not algorithms:
        return sorted(available)
    return algorithms


def _style_map(labels: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for idx, label in enumerate(labels):
        mapping[label] = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
    return mapping


def _snir_color(state: str) -> str:
    return SNIR_COLORS.get(state, SNIR_COLORS["snir_unknown"])


def _add_state_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=_snir_color("snir_off"), lw=2, label=SNIR_TITLES["snir_off"]),
        Line2D([0], [0], color=_snir_color("snir_on"), lw=2, label=SNIR_TITLES["snir_on"]),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, title="SNIR")


def _build_agg_from_raw(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    performance_rows: List[Dict[str, Any]] = []
    convergence_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    convergence_grouped: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}

    for row in decisions:
        snir_state = row.get("snir_state", "snir_unknown")
        algorithm = row.get("algorithm")
        round_idx = _parse_int(row.get("round_idx"))
        if algorithm is None or round_idx is None:
            continue
        grouped.setdefault((snir_state, algorithm, round_idx), []).append(row)
        episode_idx = _parse_int(row.get("episode_idx"))
        if episode_idx is not None:
            convergence_grouped.setdefault((snir_state, algorithm, episode_idx), []).append(row)

    for (snir_state, algorithm, round_idx), items in sorted(grouped.items()):
        reward_vals = [_parse_float(item.get("reward"), 0.0) or 0.0 for item in items]
        pdr_vals = [_parse_float(item.get("pdr"), 0.0) or 0.0 for item in items]
        throughput_vals = [_parse_float(item.get("throughput"), 0.0) or 0.0 for item in items]
        reward_mean, reward_ci = _mean_ci(reward_vals)
        pdr_mean, pdr_ci = _mean_ci(pdr_vals)
        throughput_mean, throughput_ci = _mean_ci(throughput_vals)
        performance_rows.append(
            {
                "snir_state": snir_state,
                "algorithm": algorithm,
                "round_idx": round_idx,
                "reward_mean": reward_mean,
                "reward_ci95": reward_ci,
                "pdr_mean": pdr_mean,
                "pdr_ci95": pdr_ci,
                "throughput_mean": throughput_mean,
                "throughput_ci95": throughput_ci,
            }
        )

    for (snir_state, algorithm, episode_idx), items in sorted(convergence_grouped.items()):
        reward_vals = [_parse_float(item.get("reward"), 0.0) or 0.0 for item in items]
        pdr_vals = [_parse_float(item.get("pdr"), 0.0) or 0.0 for item in items]
        throughput_vals = [_parse_float(item.get("throughput"), 0.0) or 0.0 for item in items]
        reward_mean, reward_ci = _mean_ci(reward_vals)
        pdr_mean, pdr_ci = _mean_ci(pdr_vals)
        throughput_mean, throughput_ci = _mean_ci(throughput_vals)
        convergence_rows.append(
            {
                "snir_state": snir_state,
                "algorithm": algorithm,
                "episode_idx": episode_idx,
                "reward_mean": reward_mean,
                "reward_ci95": reward_ci,
                "pdr_mean": pdr_mean,
                "pdr_ci95": pdr_ci,
                "throughput_mean": throughput_mean,
                "throughput_ci95": throughput_ci,
            }
        )

    return {"performance": performance_rows, "convergence": convergence_rows}


def _select_metrics_x_key(records: Sequence[Mapping[str, Any]]) -> str:
    candidates = ["num_nodes", "cluster", "sf", "run_id"]
    for key in candidates:
        values = {
            _parse_float(record.get(key))
            for record in records
            if record.get(key) is not None and record.get(key) != ""
        }
        if len(values) > 1:
            return key
    return "run_id"


def _x_label_for_key(key: str) -> str:
    return {
        "num_nodes": "Nombre de nœuds",
        "cluster": "Cluster",
        "sf": "SF",
        "run_id": "Run",
    }.get(key, key)


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    x_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float], List[Mapping[str, Any]]] = {}
    for row in records:
        snir_state = row.get("snir_state", "snir_unknown")
        algorithm = row.get("algorithm")
        x_value = _parse_float(row.get(x_key))
        if algorithm is None or x_value is None:
            continue
        grouped.setdefault((snir_state, algorithm, x_value), []).append(row)

    aggregated: List[Dict[str, Any]] = []
    for (snir_state, algorithm, x_value), items in sorted(grouped.items(), key=lambda item: item[0][2]):
        row: Dict[str, Any] = {
            "snir_state": snir_state,
            "algorithm": algorithm,
            "x_value": x_value,
        }
        for metric in metrics:
            values = _valid_values(_parse_float(item.get(metric)) for item in items)
            mean, ci = _mean_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        aggregated.append(row)
    return aggregated


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_performance(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
) -> None:
    metrics = [
        ("pdr_mean", "pdr_ci95", "Average PDR"),
        ("throughput_mean", "throughput_ci95", "Average throughput (bps)"),
    ]
    fig, axes = plt.subplots(nrows=len(metrics), ncols=2, figsize=(12, 4.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    style = _style_map(list(algorithms))

    for row_idx, (mean_key, ci_key, ylabel) in enumerate(metrics):
        for col_idx, state in enumerate(STATE_LABELS):
            ax = axes[row_idx][col_idx]
            ax.set_title(SNIR_TITLES.get(state, state))
            for algo in algorithms:
                subset = [
                    rec
                    for rec in records
                    if rec.get("snir_state") == state and rec.get("algorithm") == algo
                ]
                if not subset:
                    continue
                subset_sorted = sorted(subset, key=lambda rec: int(rec.get("round_idx", 0)))
                rounds = [int(rec.get("round_idx", 0)) for rec in subset_sorted]
                means = [_parse_float(rec.get(mean_key), 0.0) or 0.0 for rec in subset_sorted]
                cis = [_parse_float(rec.get(ci_key), 0.0) or 0.0 for rec in subset_sorted]
                marker = style[algo]
                ax.plot(
                    rounds,
                    means,
                    marker=marker,
                    markersize=3,
                    label=algo,
                    color=_snir_color(state),
                )
                ax.fill_between(
                    rounds,
                    [m - c for m, c in zip(means, cis)],
                    [m + c for m, c in zip(means, cis)],
                    color=_snir_color(state),
                    alpha=0.15,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.4)
            if row_idx == 0 and ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("Round")
    axes[-1][1].set_xlabel("Round")
    _add_state_legend(fig)
    _save_plot(fig, output_dir, "step2_performance_rounds.png")


def _plot_convergence(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
) -> None:
    metrics = [
        ("reward_mean", "reward_ci95", "Average reward"),
        ("pdr_mean", "pdr_ci95", "Average PDR"),
        ("throughput_mean", "throughput_ci95", "Average throughput (bps)"),
    ]
    fig, axes = plt.subplots(nrows=len(metrics), ncols=2, figsize=(12, 4.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    style = _style_map(list(algorithms))

    for row_idx, (mean_key, ci_key, ylabel) in enumerate(metrics):
        for col_idx, state in enumerate(STATE_LABELS):
            ax = axes[row_idx][col_idx]
            ax.set_title(SNIR_TITLES.get(state, state))
            for algo in algorithms:
                subset = [
                    rec
                    for rec in records
                    if rec.get("snir_state") == state and rec.get("algorithm") == algo
                ]
                if not subset:
                    continue
                subset_sorted = sorted(subset, key=lambda rec: int(rec.get("episode_idx", 0)))
                episodes = [int(rec.get("episode_idx", 0)) for rec in subset_sorted]
                means = [_parse_float(rec.get(mean_key), 0.0) or 0.0 for rec in subset_sorted]
                cis = [_parse_float(rec.get(ci_key), 0.0) or 0.0 for rec in subset_sorted]
                marker = style[algo]
                ax.plot(
                    episodes,
                    means,
                    marker=marker,
                    markersize=3,
                    label=algo,
                    color=_snir_color(state),
                )
                ax.fill_between(
                    episodes,
                    [m - c for m, c in zip(means, cis)],
                    [m + c for m, c in zip(means, cis)],
                    color=_snir_color(state),
                    alpha=0.15,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.4)
            if row_idx == 0 and ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("Episode")
    axes[-1][1].set_xlabel("Episode")
    _add_state_legend(fig)
    _save_plot(fig, output_dir, "step2_convergence_ci95.png")


def _plot_distribution(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
) -> None:
    if not records:
        return
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharey=False)
    style = _style_map(list(algorithms))

    for row_idx, state in enumerate(STATE_LABELS):
        state_records = [rec for rec in records if rec.get("snir_state") == state]
        for algo in algorithms:
            algo_records = [rec for rec in state_records if rec.get("algorithm") == algo]
            if not algo_records:
                continue
            sf_records = sorted({int(rec["sf"]) for rec in algo_records if rec.get("sf")})
            tx_records = sorted({float(rec["tx_power"]) for rec in algo_records if rec.get("tx_power")})
            marker = style[algo]
            if sf_records:
                sf_shares = [
                    sum(
                        float(rec.get("share", 0.0))
                        for rec in algo_records
                        if int(rec.get("sf", 0)) == sf
                    )
                    for sf in sf_records
                ]
                axes[row_idx][0].plot(
                    sf_records,
                    sf_shares,
                    marker=marker,
                    label=algo,
                    color=_snir_color(state),
                )
            if tx_records:
                tx_shares = [
                    sum(
                        float(rec.get("share", 0.0))
                        for rec in algo_records
                        if float(rec.get("tx_power", 0.0)) == tx
                    )
                    for tx in tx_records
                ]
                axes[row_idx][1].plot(
                    tx_records,
                    tx_shares,
                    marker=marker,
                    label=algo,
                    color=_snir_color(state),
                )

        axes[row_idx][0].set_title(f"{SNIR_TITLES.get(state, state)} - SF")
        axes[row_idx][1].set_title(f"{SNIR_TITLES.get(state, state)} - TX Power")
        axes[row_idx][0].set_ylabel("Share (0–1)")
        axes[row_idx][0].grid(True, linestyle=":", alpha=0.4)
        axes[row_idx][1].grid(True, linestyle=":", alpha=0.4)
        if axes[row_idx][0].get_legend_handles_labels()[1]:
            axes[row_idx][0].legend(fontsize=8, ncol=2)
    axes[1][0].set_xlabel("SF")
    axes[1][1].set_xlabel("TX Power")
    _add_state_legend(fig)
    _save_plot(fig, output_dir, "step2_sf_tx_distribution.png")


def _plot_metrics(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    *,
    figure_name: str,
    x_label: str,
) -> None:
    if not records:
        return
    fig, axes = plt.subplots(nrows=len(metrics), ncols=2, figsize=(12, 4.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    style = _style_map(list(algorithms))

    for row_idx, (metric_key, ylabel) in enumerate(metrics):
        mean_key = f"{metric_key}_mean"
        ci_key = f"{metric_key}_ci95"
        for col_idx, state in enumerate(STATE_LABELS):
            ax = axes[row_idx][col_idx]
            ax.set_title(SNIR_TITLES.get(state, state))
            for algo in algorithms:
                subset = [
                    rec
                    for rec in records
                    if rec.get("snir_state") == state and rec.get("algorithm") == algo
                ]
                if not subset:
                    continue
                subset_sorted = sorted(subset, key=lambda rec: float(rec.get("x_value", 0.0)))
                xs = [float(rec.get("x_value", 0.0)) for rec in subset_sorted]
                means = [_parse_float(rec.get(mean_key), 0.0) or 0.0 for rec in subset_sorted]
                cis = [_parse_float(rec.get(ci_key), 0.0) or 0.0 for rec in subset_sorted]
                marker = style[algo]
                ax.plot(xs, means, marker=marker, markersize=3, label=algo, color=_snir_color(state))
                ax.fill_between(
                    xs,
                    [m - c for m, c in zip(means, cis)],
                    [m + c for m, c in zip(means, cis)],
                    color=_snir_color(state),
                    alpha=0.15,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.4)
            if row_idx == 0 and ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=8, ncol=2)
    axes[-1][0].set_xlabel(x_label)
    axes[-1][1].set_xlabel(x_label)
    _add_state_legend(fig)
    _save_plot(fig, output_dir, figure_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant results/step2/raw et results/step2/agg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire de sortie pour les figures.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help="Limite la sélection aux algorithmes listés (répétable).",
    )
    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Ignore la figure de distribution SF/TP.",
    )
    parser.add_argument(
        "--only-core-figures",
        action="store_true",
        help="Ne génère que les figures principales (performance + convergence).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results_dir = args.results_dir
    agg_dir = results_dir / "agg"
    raw_dir = results_dir / "raw"

    performance_rows = _load_csv(agg_dir / "performance_rounds.csv")
    convergence_rows = _load_csv(agg_dir / "convergence.csv")
    if not performance_rows or not convergence_rows:
        decision_rows = _load_csv(raw_dir / "decisions.csv")
        if decision_rows:
            generated = _build_agg_from_raw(decision_rows)
            performance_rows = performance_rows or generated["performance"]
            convergence_rows = convergence_rows or generated["convergence"]

    metrics_rows = _load_csv(raw_dir / "metrics.csv")

    algorithms = _select_algorithms(
        performance_rows or convergence_rows or metrics_rows,
        args.algorithm or None,
    )
    if performance_rows:
        _plot_performance(performance_rows, args.output_dir, algorithms)
    if convergence_rows:
        _plot_convergence(convergence_rows, args.output_dir, algorithms)

    if metrics_rows:
        x_key = _select_metrics_x_key(metrics_rows)
        x_label = _x_label_for_key(x_key)
        metric_keys = ["der", "snir_avg", "energy_j", "fairness"]
        aggregated_metrics = _aggregate_metrics(metrics_rows, metric_keys + ["pdr"], x_key)
        _plot_metrics(
            aggregated_metrics,
            args.output_dir,
            algorithms,
            [
                ("der", "DER"),
                ("snir_avg", "SNIR moyen"),
                ("energy_j", "Énergie (J)"),
                ("fairness", "Équité"),
            ],
            figure_name="step2_metrics_ci95.png",
            x_label=x_label,
        )
        _plot_metrics(
            aggregated_metrics,
            args.output_dir,
            algorithms,
            [
                ("pdr", "PDR"),
                ("der", "DER"),
                ("snir_avg", "SNIR moyen"),
                ("energy_j", "Énergie (J)"),
                ("fairness", "Équité"),
            ],
            figure_name="step2_key_metrics_combined.png",
            x_label=x_label,
        )

    if not args.skip_distribution and not args.only_core_figures:
        distribution_rows = _load_csv(agg_dir / "sf_tp_distribution.csv")
        if distribution_rows:
            _plot_distribution(distribution_rows, args.output_dir, algorithms)


if __name__ == "__main__":
    main()
