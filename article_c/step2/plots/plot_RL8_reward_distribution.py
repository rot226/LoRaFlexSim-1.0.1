"""Trace la figure RL8 (distribution des récompenses par algorithme)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    filter_rows_by_network_sizes,
    filter_cluster,
    normalize_network_size_rows,
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


def _title_suffix(network_sizes: list[int]) -> str:
    if len(network_sizes) == 1:
        return " (taille unique)"
    return ""


def _load_step2_raw_results(
    results_path: Path,
    *,
    allow_sample: bool = True,
) -> list[dict[str, object]]:
    if not results_path.exists():
        if allow_sample:
            return []
        raise FileNotFoundError(f"CSV Step2 manquant: {results_path}")
    df = pd.read_csv(results_path)
    if df.empty:
        return []
    if "reward" not in df.columns:
        warnings.warn(
            "Colonne reward manquante dans raw_results.csv; figure ignorée.",
            stacklevel=2,
        )
        return []
    if "network_size" in df.columns:
        network_size_series = df["network_size"]
    elif "density" in df.columns:
        network_size_series = df["density"]
    else:
        network_size_series = pd.Series([None] * len(df))
    df["network_size"] = pd.to_numeric(network_size_series, errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df["algo"] = df.get("algo", "")
    df["snir_mode"] = df.get("snir_mode", "")
    df["cluster"] = df.get("cluster", "all").fillna("all")
    df = df.dropna(subset=["network_size", "reward"])
    return df.to_dict(orient="records")


def _plot_distribution(
    rows: list[dict[str, object]],
    network_sizes: list[int],
) -> plt.Figure:
    fig, ax = plt.subplots()
    algorithms = sorted({row["algo"] for row in rows})
    rewards_by_algo = [
        [row["reward"] for row in rows if row["algo"] == algo]
        for algo in algorithms
    ]
    positions = list(range(1, len(algorithms) + 1))
    ax.violinplot(rewards_by_algo, positions=positions, showmedians=True)
    ax.boxplot(
        rewards_by_algo,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.6},
        medianprops={"color": "black"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([algo_label(str(algo)) for algo in algorithms])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Reward")
    ax.set_title(
        "Step 2 - Reward Distribution by Algorithm"
        f"{_title_suffix(network_sizes)}"
    )
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
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "raw_results.csv"
    rows = _load_step2_raw_results(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)
    if network_sizes_filter is None:
        df = pd.DataFrame(rows)
        network_sizes = sorted(df["network_size"].unique())
    else:
        network_sizes = network_sizes_filter
    if _has_invalid_network_sizes(network_sizes):
        return
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )

    fig = _plot_distribution(rows, network_sizes)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL8_reward_distribution", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
