"""Plot class density metrics for the MNE3SD article A analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.mne3sd.common import prepare_figure_directory, save_figure

ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "class_density_metrics.csv"
ARTICLE = "article_a"
SCENARIO = "class_density"


def apply_plot_style(style: str | None) -> None:
    """Apply a clean default plotting style unless a custom one is provided."""
    plt.rcdefaults()
    if style:
        plt.style.use(style)
        return

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.figsize": (3.5, 2.2),
        }
    )


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate node density plots showing packet delivery ratio and "
            "optionally per-node energy consumption for LoRaWAN classes."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the class_density_metrics.csv file",
    )
    parser.add_argument(
        "--style",
        help="Matplotlib style name or .mplstyle path to override the default settings",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures instead of running in batch mode",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    """Read the metrics CSV, ensuring mandatory columns are present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}. Run the class density sweep first."
        )

    df = pd.read_csv(path)
    required = {"class", "nodes", "replicate", "pdr"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["class"] = df["class"].astype(str)
    df["nodes"] = df["nodes"].astype(int)
    df["replicate"] = df["replicate"].astype(int)
    df["pdr"] = df["pdr"].astype(float)

    if "energy_per_node_J" in df.columns:
        df["energy_per_node_J"] = pd.to_numeric(df["energy_per_node_J"], errors="coerce")

    return df


def summarise_metric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return mean and standard deviation of ``column`` per class/node pair."""
    summary = (
        df.groupby(["class", "nodes"], as_index=False)[column]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.rename(columns={"mean": f"{column}_mean", "std": f"{column}_std"}, inplace=True)
    summary[f"{column}_std"] = summary[f"{column}_std"].fillna(0.0)
    return summary.sort_values(["class", "nodes"])  # keep ordering deterministic


def plot_pdr_vs_nodes(df: pd.DataFrame) -> None:
    """Plot the packet delivery ratio versus node count for each class."""
    stats = summarise_metric(df, "pdr")
    stats["pdr_mean"] *= 100.0
    stats["pdr_std"] *= 100.0

    fig, ax = plt.subplots()

    for class_name, class_data in stats.groupby("class"):
        ordered = class_data.sort_values("nodes")
        ax.errorbar(
            ordered["nodes"],
            ordered["pdr_mean"],
            yerr=ordered["pdr_std"],
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Packet delivery ratio (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Class")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="pdr_vs_nodes",
    )
    save_figure(fig, "class_pdr_vs_nodes", output_dir)


def plot_energy_vs_nodes(df: pd.DataFrame) -> bool:
    """Plot average per-node energy versus node count if data is available."""
    if "energy_per_node_J" not in df.columns:
        return False

    if df["energy_per_node_J"].dropna().empty:
        return False

    stats = summarise_metric(df, "energy_per_node_J")

    fig, ax = plt.subplots()

    for class_name, class_data in stats.groupby("class"):
        ordered = class_data.sort_values("nodes")
        ax.errorbar(
            ordered["nodes"],
            ordered["energy_per_node_J_mean"],
            yerr=ordered["energy_per_node_J_std"],
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Energy per node (J)")
    ax.legend(title="Class")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_vs_nodes",
    )
    save_figure(fig, "class_energy_vs_nodes", output_dir)
    return True


def main() -> None:
    args = parse_arguments()

    apply_plot_style(args.style)

    metrics = load_metrics(args.results)

    plot_pdr_vs_nodes(metrics)
    energy_created = plot_energy_vs_nodes(metrics)

    if not energy_created:
        print("Energy per node data not available; skipping energy plot.")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
