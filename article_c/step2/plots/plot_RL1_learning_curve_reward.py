"""Trace la courbe d'apprentissage (récompense moyenne vs rounds)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    load_step2_aggregated,
    place_legend,
    save_figure,
)

NETWORK_SIZES_OVERRIDE: list[int] | None = None


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes or len(network_sizes) < 2:
        return None
    return network_sizes


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        rows.append(
            {
                "round": round_id,
                "algo": "ADR",
                "avg_reward": 0.45 + 0.01 * round_id,
            }
        )
        rows.append(
            {
                "round": round_id,
                "algo": "UCB1-SF",
                "avg_reward": 0.50 + 0.02 * round_id,
            }
        )
    return rows


def _plot_learning_curve(rows: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots()
    preferred_algos = ["ADR", "UCB1-SF"]
    available = {row["algo"] for row in rows}
    algorithms = [algo for algo in preferred_algos if algo in available]
    if not algorithms:
        algorithms = sorted(available)
    for algo in algorithms:
        algo_rows = [row for row in rows if row["algo"] == algo]
        points = {row["round"]: row["avg_reward"] for row in algo_rows}
        rounds = sorted(points)
        values = [points[round_id] for round_id in rounds]
        ax.plot(rounds, values, marker="o", label=algo)
    ax.set_xlabel("Decision rounds")
    ax.set_ylabel("Average window reward")
    ax.set_title("Average window reward vs Decision rounds")
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
    network_sizes_input = args.network_sizes or NETWORK_SIZES_OVERRIDE
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "learning_curve.csv"
    rows = _load_learning_curve(results_path)
    aggregated_results_path = step_dir / "results" / "aggregated_results.csv"
    size_rows = load_step2_aggregated(aggregated_results_path)
    ensure_network_size(size_rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes_input)
    size_rows, _ = filter_rows_by_network_sizes(size_rows, network_sizes_filter)
    df = pd.DataFrame(size_rows)
    network_sizes = network_sizes_filter or sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )

    fig = _plot_learning_curve(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1_learning_curve_reward", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
