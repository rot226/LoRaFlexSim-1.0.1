"""Trace un nuage de points récompense moyenne vs PDR agrégé."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    ensure_network_size,
    filter_cluster,
    filter_rows_by_network_sizes,
    load_step1_aggregated,
    load_step2_aggregated,
    place_legend,
    save_figure,
)

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
    if not network_sizes or len(network_sizes) < 2:
        return None
    return network_sizes


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_learning_curve_means(path: Path) -> dict[str, float]:
    rows = _load_learning_curve(path)
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


def _load_learning_curve(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return _sample_learning_curve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return _sample_learning_curve()
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
) -> dict[str, float]:
    rows = filter_cluster(load_step1_aggregated(path), "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
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
) -> dict[str, float]:
    rows = filter_cluster(load_step2_aggregated(path), "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
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
) -> list[dict[str, float | str]]:
    reward_means = _load_learning_curve_means(learning_curve_path)
    pdr_means = _aggregate_pdr_from_step1(step1_results_path, network_sizes)
    missing = [algo for algo in reward_means if algo not in pdr_means]
    if missing:
        step2_pdr = _aggregate_pdr_from_step2(step2_results_path, network_sizes)
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


def _plot_scatter(points: list[dict[str, float | str]]) -> plt.Figure:
    fig, ax = plt.subplots()
    for point in points:
        algo = str(point["algo"])
        ax.scatter(point["pdr_mean"], point["reward_mean"], label=algo_label(algo))
    ax.set_xlabel("Aggregated PDR (probability)")
    ax.set_ylabel("Mean window reward")
    ax.set_title("Step 2 - Mean reward vs Aggregated PDR")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    place_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args()
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    learning_curve_path = step_dir / "results" / "learning_curve.csv"
    step1_results_path = step_dir.parents[1] / "step1" / "results" / "aggregated_results.csv"
    step2_results_path = step_dir / "results" / "aggregated_results.csv"
    size_rows = load_step2_aggregated(step2_results_path)
    ensure_network_size(size_rows)
    network_sizes_filter = _normalized_network_sizes(args.network_sizes)
    size_rows, _ = filter_rows_by_network_sizes(size_rows, network_sizes_filter)
    df = pd.DataFrame(size_rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    points = _collect_points(
        learning_curve_path,
        step1_results_path,
        step2_results_path,
        network_sizes_filter,
    )

    fig = _plot_scatter(points)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL10_reward_vs_pdr_scatter", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
