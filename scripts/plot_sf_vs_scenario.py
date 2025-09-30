#!/usr/bin/env python3
"""Plot average spreading factor by scenario or model.

Usage::

    python scripts/plot_sf_vs_scenario.py results/mobility_latency_energy.csv
    python scripts/plot_sf_vs_scenario.py --by-model results/mobility_models.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: str, output_dir: str = "figures", by_model: bool = False) -> None:
    """Plot average spreading factor with error bars.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the ``avg_sf_mean`` and ``avg_sf_std``
        columns along with either ``scenario`` or ``model``.
    output_dir:
        Directory where the figure will be written.
    by_model:
        If ``True`` plot against the ``model`` column, otherwise use
        ``scenario``.
    """
    df = pd.read_csv(csv_path)

    x_col = "model" if by_model else "scenario"
    required = {x_col, "avg_sf_mean", "avg_sf_std"}
    if not required <= set(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise SystemExit(f"CSV must contain columns: {missing}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(df[x_col]))
    bars = ax.bar(
        x,
        df["avg_sf_mean"],
        yerr=df["avg_sf_std"],
        capsize=4,
        color="C0",
        label="Average SF",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=45, ha="right")
    ax.set_xlabel("Mobility model" if by_model else "Scenario")
    ax.set_ylabel("Average SF")
    ax.set_title("Average SF by " + ("model" if by_model else "scenario"))
    ax.bar_label(bars, fmt="%.2f", label_type="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    stem = "avg_sf_vs_model" if by_model else "avg_sf_vs_scenario"
    for ext in ("png", "jpg", "eps"):
        dpi = 300 if ext in ("png", "jpg", "eps") else None
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to CSV file with average SF metrics")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        help="Plot average SF versus mobility model",
    )
    args = parser.parse_args(argv)
    plot(args.csv, args.output_dir, args.by_model)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
