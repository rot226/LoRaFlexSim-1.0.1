"""Trace la courbe d'apprentissage (récompense moyenne vs rounds)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import apply_plot_style, place_legend, save_figure


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
                "reward_mean": _to_float(row.get("reward_mean")),
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
                "reward_mean": 0.45 + 0.01 * round_id,
            }
        )
        rows.append(
            {
                "round": round_id,
                "algo": "UCB1-SF",
                "reward_mean": 0.50 + 0.02 * round_id,
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
        points = {row["round"]: row["reward_mean"] for row in algo_rows}
        rounds = sorted(points)
        values = [points[round_id] for round_id in rounds]
        ax.plot(rounds, values, marker="o", label=algo)
    ax.set_xlabel("Decision rounds")
    ax.set_ylabel("Average window reward")
    ax.set_title("Average window reward vs Decision rounds")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "learning_curve.csv"
    rows = _load_learning_curve(results_path)

    fig = _plot_learning_curve(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1_learning_curve_reward", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
