#!/usr/bin/env python3
"""Plot PDR, latency and energy metrics from mobility_latency_energy.csv.

Usage::

    python scripts/plot_mobility_latency_energy.py results/mobility_latency_energy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(
    csv_path: str,
    output_dir: str = "figures",
    max_delay: float | None = None,
    max_energy: float | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    if "nodes" in df.columns:
        df["scenario_label"] = (
            df["scenario"] + " (" + df["nodes"].astype(str) + " nodes)"
        )
    else:
        df["scenario_label"] = df["scenario"]
    plt.rcParams.update({"font.size": 16})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = []
    if "nodes" in df.columns:
        params.append(f"nodes={int(df['nodes'].iloc[0])}")
    if "interval" in df.columns:
        params.append(f"interval={df['interval'].iloc[0]:g}s")
    if "speed" in df.columns:
        params.append(f"speed={df['speed'].iloc[0]:g}m/s")
    if "area_size" in df.columns:
        params.append(f"area={df['area_size'].iloc[0] ** 2:g}m²")
    if "channels" in df.columns:
        params.append(f"channels={int(df['channels'].iloc[0])}")
    param_text = ", ".join(params)

    metrics = [
        ("pdr", "PDR", "%", "%.1f%%", "C0", "pdr_vs_scenario.svg"),
        (
            "avg_delay",
            "Average delay",
            "s",
            "%.2f s",
            "C1",
            "avg_delay_vs_scenario.svg",
        ),
        (
            "energy_per_node",
            "Average energy per node",
            "J",
            "%.2f J",
            "C2",
            "avg_energy_per_node_vs_scenario.svg",
        ),
        (
            "avg_sf",
            "Average SF",
            "SF",
            "%.1f",
            "C4",
            "avg_sf_vs_scenario.svg",
        ),
    ]

    for metric, name, unit, fmt, color, filename in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        yerr = df[std_col] if std_col in df.columns else None
        fig, ax = plt.subplots(figsize=(16, 8))
        label = f"{name} ({unit})"
        bars = ax.bar(
            df["scenario"],
            df[mean_col],
            yerr=yerr,
            capsize=4,
            color=color,
            label=label,
        )
        ax.set_xlabel("")
        ax.set_xticks(range(len(df["scenario"])))
        ax.set_xticklabels(df["scenario_label"], rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.tick_params(axis="both", labelsize=16)

        if metric == "pdr":
            cap = 100.0
            ax.set_ylim(0, cap)
            ax.axhline(cap, linestyle="--", color="grey")
        elif metric == "avg_delay":
            cap = max_delay or df[mean_col].max() * 1.1
            ax.set_ylim(0, cap)
        elif metric == "energy_per_node":
            cap = max_energy or df[mean_col].max() * 1.1
            ax.set_ylim(0, cap)
        else:
            cap = df[mean_col].max() * 1.1
            ax.set_ylim(0, cap)

        ax.bar_label(bars, fmt=fmt, label_type="center")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),
            ncol=1,
            title="N: number of nodes, C: number of channels, speed: m/s",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.85])
        stem = Path(filename).stem
        for ext in ("png", "jpg", "eps", "svg"):
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
    parser.add_argument("csv", help="Path to mobility_latency_energy.csv")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=None,
        help="Y-axis maximum for average delay plots",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=None,
        help="Y-axis maximum for energy plots",
    )
    args = parser.parse_args(argv)
    plot(args.csv, args.output_dir, args.max_delay, args.max_energy)


if __name__ == "__main__":
    main()
