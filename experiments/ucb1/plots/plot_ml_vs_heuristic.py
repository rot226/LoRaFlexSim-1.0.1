"""Comparaison ML vs heuristique (PDR, énergie, fairness)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DECISION_CSV = Path(__file__).resolve().parents[1] / "ucb1_baseline_decision_log.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots"
POLICY_LABELS = {"ml": "ML (UCB1)", "heuristic": "Heuristique"}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparaison ML vs heuristiques.")
    parser.add_argument(
        "--decision-csv",
        type=Path,
        default=DEFAULT_DECISION_CSV,
        help="CSV des décisions (run_baseline_comparison.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les PNG.",
    )
    parser.add_argument(
        "--fairness-window",
        type=int,
        default=200,
        help="Taille de la fenêtre (en décisions) pour la fairness temporelle.",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(missing)}")


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _jain_fairness(values: list[float]) -> float:
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value ** 2 for value in values)
    return numerator / denominator if denominator > 0 else 0.0


def _resolve_time_column(df: pd.DataFrame) -> str:
    if "time_s" in df.columns:
        return "time_s"
    return "decision_idx"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.decision_csv)
    _ensure_columns(
        df,
        ["policy", "num_nodes", "pdr", "energy_j", "throughput", "cluster", "decision_idx"],
        args.decision_csv,
    )
    time_col = _resolve_time_column(df)

    df = df.sort_values(time_col)
    policies = sorted(df["policy"].dropna().unique())
    policy_colors = {policy: PALETTE[idx % len(PALETTE)] for idx, policy in enumerate(policies)}

    final_pdr = (
        df.groupby(["policy", "num_nodes", "cluster"], as_index=False)["pdr"]
        .last()
        .groupby(["policy", "num_nodes"], as_index=False)["pdr"]
        .mean()
    )

    energy_summary = (
        df.groupby(["policy", "num_nodes", "cluster"], as_index=False)["energy_j"]
        .sum()
        .groupby(["policy", "num_nodes"], as_index=False)["energy_j"]
        .mean()
    )

    energy_vs_pdr = energy_summary.merge(final_pdr, on=["policy", "num_nodes"], how="inner")

    df["fairness_bin"] = (df["decision_idx"] // args.fairness_window) * args.fairness_window
    fairness_rows: list[dict[str, float]] = []
    for (policy, bin_id), group in df.groupby(["policy", "fairness_bin"]):
        throughput_by_cluster = (
            group.groupby("cluster", as_index=False)["throughput"].mean()["throughput"].tolist()
        )
        fairness_rows.append(
            {
                "policy": policy,
                "fairness_bin": bin_id,
                "fairness": _jain_fairness(throughput_by_cluster),
            }
        )
    fairness_df = pd.DataFrame(fairness_rows)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 10.5))

    ax = axes[0]
    for policy in policies:
        subset = final_pdr[final_pdr["policy"] == policy]
        if subset.empty:
            continue
        ax.plot(
            subset["num_nodes"],
            subset["pdr"],
            marker="o",
            label=POLICY_LABELS.get(policy, policy),
            color=policy_colors[policy],
        )
    ax.set_title("PDR vs densité (num_nodes)")
    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("PDR moyenne")
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for policy in policies:
        subset = energy_vs_pdr[energy_vs_pdr["policy"] == policy]
        if subset.empty:
            continue
        ax.scatter(
            subset["pdr"],
            subset["energy_j"],
            label=POLICY_LABELS.get(policy, policy),
            color=policy_colors[policy],
        )
    ax.set_title("Énergie vs PDR")
    ax.set_xlabel("PDR moyenne")
    ax.set_ylabel("Énergie moyenne par cluster (J)")
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)

    ax = axes[2]
    for policy in policies:
        subset = fairness_df[fairness_df["policy"] == policy]
        if subset.empty:
            continue
        ax.plot(
            subset["fairness_bin"],
            subset["fairness"],
            label=POLICY_LABELS.get(policy, policy),
            color=policy_colors[policy],
            marker="o",
            markersize=3,
        )
    ax.set_title("Fairness Jain vs temps (fenêtres de décisions)")
    ax.set_xlabel("Indice de décision (fenêtre)")
    ax.set_ylabel("Indice de fairness")
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)

    _save_plot(fig, args.output_dir, "ucb1_ml_vs_heuristic.png")


if __name__ == "__main__":
    main()
