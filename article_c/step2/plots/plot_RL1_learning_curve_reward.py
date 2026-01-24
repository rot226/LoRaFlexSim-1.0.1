"""Trace la courbe d'apprentissage (récompense moyenne vs rounds)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from article_c.common.plot_helpers import (
    apply_plot_style,
    apply_figure_layout,
    filter_rows_by_network_sizes,
    place_legend,
    save_figure,
)


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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_learning_curve(
    path: Path,
    *,
    allow_sample: bool = True,
) -> list[dict[str, object]]:
    if not path.exists():
        if allow_sample:
            return _sample_learning_curve()
        warnings.warn(
            f"CSV introuvable ({path}). Fallback échantillon désactivé.",
            stacklevel=2,
        )
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        if allow_sample:
            return _sample_learning_curve()
        warnings.warn(
            f"CSV vide ({path}). Fallback échantillon désactivé.",
            stacklevel=2,
        )
        return []
    parsed: list[dict[str, object]] = []
    for row in rows:
        network_size_value = row.get("network_size")
        if network_size_value in (None, "") and "density" in row:
            network_size_value = row.get("density")
        parsed.append(
            {
                "round": int(_to_float(row.get("round"))),
                "algo": row.get("algo", ""),
                "avg_reward": _to_float(row.get("avg_reward")),
                "network_size": _to_float(network_size_value),
            }
        )
    return parsed


def _sample_learning_curve() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_id in range(10):
        for network_size in (50, 100):
            rows.append(
                {
                    "round": round_id,
                    "algo": "ADR",
                    "avg_reward": 0.45 + 0.01 * round_id - 0.0005 * network_size,
                    "network_size": network_size,
                }
            )
            rows.append(
                {
                    "round": round_id,
                    "algo": "UCB1-SF",
                    "avg_reward": 0.50 + 0.02 * round_id - 0.0003 * network_size,
                    "network_size": network_size,
                }
            )
    return rows


def _aggregate_reward_by_round(
    rows: list[dict[str, object]],
) -> dict[int, float]:
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        round_id = int(_to_float(row.get("round")))
        reward = _to_float(row.get("avg_reward"))
        totals[round_id] = totals.get(round_id, 0.0) + reward
        counts[round_id] = counts.get(round_id, 0) + 1
    return {
        round_id: totals[round_id] / counts[round_id]
        for round_id in totals
        if counts.get(round_id, 0) > 0
    }


def _plot_learning_curve(
    rows: list[dict[str, object]],
    *,
    reference_size: int | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots()
    width, height = fig.get_size_inches()
    apply_figure_layout(fig, figsize=(width, height + 2))
    preferred_algos = ["ADR", "UCB1-SF"]
    available = {row["algo"] for row in rows}
    algorithms = [algo for algo in preferred_algos if algo in available]
    if not algorithms:
        algorithms = sorted(available)
    network_sizes = sorted(
        {int(_to_float(row.get("network_size"))) for row in rows}
    )
    if reference_size is not None or len(network_sizes) <= 1:
        for algo in algorithms:
            algo_rows = [
                row
                for row in rows
                if row["algo"] == algo
                and (reference_size is None or row["network_size"] == reference_size)
            ]
            points = _aggregate_reward_by_round(algo_rows)
            if not points:
                continue
            rounds = sorted(points)
            values = [points[round_id] for round_id in rounds]
            ax.plot(rounds, values, marker="o", label=algo)
    else:
        for network_size in network_sizes:
            size_rows = [
                row for row in rows if row.get("network_size") == network_size
            ]
            points = _aggregate_reward_by_round(size_rows)
            if not points:
                continue
            rounds = sorted(points)
            values = [points[round_id] for round_id in rounds]
            ax.plot(rounds, values, marker="o", label=f"Taille {network_size}")
    ax.set_xlabel("Decision rounds")
    ax.set_ylabel("Average window reward")
    ax.set_title("Average window reward vs Decision rounds")
    place_legend(ax)
    return fig


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        help=(
            "Taille de réseau de référence à tracer "
            "(ex: --reference-size 100)."
        ),
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    reference_size = args.reference_size
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "learning_curve.csv"
    rows = _load_learning_curve(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, available_network_sizes = filter_rows_by_network_sizes(
        rows,
        network_sizes_filter,
    )
    if network_sizes_filter is None:
        network_sizes = available_network_sizes
    else:
        network_sizes = network_sizes_filter
    if _has_invalid_network_sizes(network_sizes):
        return
    if reference_size is not None:
        if _has_invalid_network_sizes([reference_size]):
            return
        if reference_size not in available_network_sizes:
            warnings.warn(
                "Taille de référence absente. Tailles disponibles: "
                + ", ".join(str(size) for size in available_network_sizes),
                stacklevel=2,
            )
            return
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    if reference_size is not None:
        rows = [row for row in rows if row.get("network_size") == reference_size]
        if not rows:
            warnings.warn(
                "Aucune donnée pour la taille de référence demandée.",
                stacklevel=2,
            )
            return

    fig = _plot_learning_curve(rows, reference_size=reference_size)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1_learning_curve_reward", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
