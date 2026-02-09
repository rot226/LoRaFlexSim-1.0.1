"""Trace un nuage de points récompense moyenne vs PDR agrégé."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from article_c.common.config import BASE_DIR
from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    MetricStatus,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    filter_cluster,
    filter_rows_by_network_sizes,
    is_constant_metric,
    load_step1_aggregated,
    load_step2_aggregated,
    normalize_network_size_rows,
    place_adaptive_legend,
    render_metric_status,
    save_figure,
)
from plot_defaults import RL_FIGURE_SCALE, resolve_ieee_figsize

ALGO_ALIASES = {
    "adr": "adr",
    "ADR": "adr",
    "mixra_h": "mixra_h",
    "MixRA-H": "mixra_h",
    "mixra_opt": "mixra_opt",
    "MixRA-Opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "UCB1-SF": "ucb1_sf",
}
TARGET_ALGOS = ("ucb1_sf", "adr", "mixra_h", "mixra_opt")


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes:
        return None
    return network_sizes


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        print(
            "ERREUR: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée."
        )
        return True
    return False


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_learning_curve_means(
    path: Path,
    *,
    allow_sample: bool = True,
) -> dict[str, float]:
    rows = _load_learning_curve(path, allow_sample=allow_sample)
    if not rows:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        algo = _canonical_algo(str(row.get("algo", "")))
        if algo is None:
            continue
        totals[algo] = totals.get(algo, 0.0) + _to_float(row.get("avg_reward"))
        counts[algo] = counts.get(algo, 0) + 1
    return {
        algo: totals[algo] / counts[algo]
        for algo in totals
        if counts.get(algo, 0) > 0
    }


def _load_learning_curve(
    path: Path,
    *,
    allow_sample: bool = True,
) -> list[dict[str, object]]:
    if not path.exists():
        if allow_sample:
            return _sample_learning_curve()
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        if allow_sample:
            return _sample_learning_curve()
        return []
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "round": int(_to_float(row.get("round"))),
                "algo": row.get("algo", ""),
                "avg_reward": _to_float(row.get("avg_reward")),
            }
        )
    return parsed


def _sample_learning_curve() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_id in range(10):
        rows.extend(
            [
                {
                    "round": round_id,
                    "algo": "ADR",
                    "avg_reward": 0.45 + 0.01 * round_id,
                },
                {
                    "round": round_id,
                    "algo": "MixRA-H",
                    "avg_reward": 0.48 + 0.012 * round_id,
                },
                {
                    "round": round_id,
                    "algo": "MixRA-Opt",
                    "avg_reward": 0.50 + 0.013 * round_id,
                },
                {
                    "round": round_id,
                    "algo": "UCB1-SF",
                    "avg_reward": 0.52 + 0.02 * round_id,
                },
            ]
        )
    return rows


def _aggregate_pdr_from_step1(
    path: Path,
    network_sizes: list[int] | None,
    *,
    allow_sample: bool = True,
) -> dict[str, float]:
    rows = filter_cluster(
        load_step1_aggregated(path, allow_sample=allow_sample),
        "all",
    )
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        algo = _canonical_algo(str(row.get("algo", "")))
        if algo is None:
            continue
        pdr = row.get("pdr_mean")
        if pdr is None:
            continue
        totals[algo] = totals.get(algo, 0.0) + float(pdr)
        counts[algo] = counts.get(algo, 0) + 1
    return {
        algo: totals[algo] / counts[algo]
        for algo in totals
        if counts.get(algo, 0) > 0
    }


def _aggregate_pdr_from_step2(
    path: Path,
    network_sizes: list[int] | None,
    *,
    allow_sample: bool = True,
) -> dict[str, float]:
    rows = filter_cluster(
        load_step2_aggregated(path, allow_sample=allow_sample),
        "all",
    )
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        algo = _canonical_algo(str(row.get("algo", "")))
        if algo is None:
            continue
        success_rate = row.get("success_rate_mean")
        if success_rate is None:
            continue
        totals[algo] = totals.get(algo, 0.0) + float(success_rate)
        counts[algo] = counts.get(algo, 0) + 1
    return {
        algo: totals[algo] / counts[algo]
        for algo in totals
        if counts.get(algo, 0) > 0
    }


def _collect_points(
    learning_curve_path: Path,
    step1_results_path: Path,
    step2_results_path: Path,
    network_sizes: list[int] | None,
    *,
    allow_sample: bool = True,
) -> list[dict[str, float | str]]:
    reward_means = _load_learning_curve_means(
        learning_curve_path,
        allow_sample=allow_sample,
    )
    pdr_means = _aggregate_pdr_from_step1(
        step1_results_path,
        network_sizes,
        allow_sample=allow_sample,
    )
    missing = [algo for algo in reward_means if algo not in pdr_means]
    if missing:
        step2_pdr = _aggregate_pdr_from_step2(
            step2_results_path,
            network_sizes,
            allow_sample=allow_sample,
        )
        for algo in missing:
            if algo in step2_pdr:
                pdr_means[algo] = step2_pdr[algo]
    points: list[dict[str, float | str]] = []
    for algo in TARGET_ALGOS:
        if algo not in reward_means or algo not in pdr_means:
            continue
        points.append(
            {
                "algo": algo,
                "reward_mean": reward_means[algo],
                "pdr_mean": pdr_means[algo],
            }
        )
    return points


def _legend_handles_for_algos(
    algos: list[str],
) -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for algo in algos:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ALGO_COLORS.get(algo, "#333333"),
                marker=ALGO_MARKERS.get(algo, "o"),
                linestyle="none",
                markersize=6.0,
            )
        )
        labels.append(algo_label(algo))
    return handles, labels


def _plot_scatter(points: list[dict[str, float | str]]) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=resolve_ieee_figsize(len(points), scale=RL_FIGURE_SCALE)
    )
    reward_values = [
        float(point["reward_mean"])
        for point in points
        if isinstance(point.get("reward_mean"), (int, float))
    ]
    pdr_values = [
        float(point["pdr_mean"])
        for point in points
        if isinstance(point.get("pdr_mean"), (int, float))
    ]
    reward_state = is_constant_metric(reward_values)
    pdr_state = is_constant_metric(pdr_values)
    if reward_state is not MetricStatus.OK or pdr_state is not MetricStatus.OK:
        metric_state = (
            MetricStatus.MISSING
            if MetricStatus.MISSING in (reward_state, pdr_state)
            else MetricStatus.CONSTANT
        )
        algos = [str(point["algo"]) for point in points]
        render_metric_status(
            fig,
            ax,
            metric_state,
            legend_loc="right",
            show_fallback_legend=True,
            legend_handles=_legend_handles_for_algos(algos),
        )
        return fig
    for point in points:
        algo = str(point["algo"])
        ax.scatter(
            point["pdr_mean"],
            point["reward_mean"],
            label=algo_label(algo),
            color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "o"),
        )
    ax.set_xlabel("Aggregated PDR (prob.)")
    ax.set_ylabel("Mean reward (a.u.)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    place_adaptive_legend(fig, ax, preferred_loc="right")
    return fig


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    step_dir = Path(__file__).resolve().parents[1]
    learning_curve_path = step_dir / "results" / "learning_curve.csv"
    step1_results_path = BASE_DIR / "step1" / "results" / "aggregated_results.csv"
    step2_results_path = step_dir / "results" / "aggregated_results.csv"
    if not step1_results_path.exists():
        step1_results_dir = BASE_DIR / "step1" / "results"
        print("INFO: CSV Step1 manquant, plot RL10 ignoré.")
        print(f"INFO: chemin attendu: {step1_results_path}")
        print(f"INFO: chemin réel possible: {step1_results_dir}")
        print("INFO: suggestion: python -m article_c.run_all --skip-step2 ...")
        return
    step1_rows = load_step1_aggregated(
        step1_results_path,
        allow_sample=allow_sample,
    )
    step2_rows = load_step2_aggregated(
        step2_results_path,
        allow_sample=allow_sample,
    )
    if not step1_rows or not step2_rows:
        print("INFO: CSV Step1/Step2 manquant ou vide, plot RL10 ignoré.")
        return
    normalize_network_size_rows(step1_rows)
    normalize_network_size_rows(step2_rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    step1_rows, _ = filter_rows_by_network_sizes(step1_rows, network_sizes_filter)
    step2_rows, _ = filter_rows_by_network_sizes(step2_rows, network_sizes_filter)
    if network_sizes_filter is None:
        step1_df = pd.DataFrame(step1_rows)
        step2_df = pd.DataFrame(step2_rows)
        step1_sizes = (
            sorted(step1_df["network_size"].unique()) if not step1_df.empty else []
        )
        step2_sizes = (
            sorted(step2_df["network_size"].unique()) if not step2_df.empty else []
        )
    else:
        step1_sizes = network_sizes_filter
        step2_sizes = network_sizes_filter
    network_sizes = sorted(set(step1_sizes) & set(step2_sizes))
    if _has_invalid_network_sizes(network_sizes):
        return
    if len(network_sizes) < 2:
        print(
            "INFO: moins de deux tailles de réseau communes entre step1 et step2 "
            f"({network_sizes}). Le plot RL10 est ignoré."
        )
        return
    points = _collect_points(
        learning_curve_path,
        step1_results_path,
        step2_results_path,
        network_sizes,
        allow_sample=allow_sample,
    )
    if not points:
        print("INFO: données insuffisantes, plot RL10 ignoré.")
        return

    fig = _plot_scatter(points)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL10_reward_vs_pdr_scatter", use_tight=False)
    assert_legend_present(fig, "plot_RL10_reward_vs_pdr_scatter")
    plt.close(fig)


if __name__ == "__main__":
    main()
