#!/usr/bin/env python3
"""Plot metrics from an aggregated mobility multichannel simulation.

Usage::

    python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv
    python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv \
        --allowed 50,1 200,3
    python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv \
        --scenarios n50_c1_static n50_c1_mobile n50_c3_mobile n50_c6_static \
        n200_c1_static n200_c1_mobile n200_c3_mobile n200_c6_static
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
    formats: tuple[str, ...] = ("png", "jpg", "svg", "eps"),
    allowed: set[tuple[int, int]] | None = None,
    scenarios: set[str] | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    if hasattr(plt, "rcParams"):
        plt.rcParams.update({"font.size": 16})

    if "scenario" not in df.columns:
        raise ValueError("CSV must contain a 'scenario' column")

    if scenarios is not None:
        df = df[df["scenario"].isin(scenarios)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df["scenario_label"] = df.apply(
        lambda r: f"N={int(r['nodes'])}, C={int(r['channels'])}, static"
        if not r["mobility"]
        else f"N={int(r['nodes'])}, C={int(r['channels'])}, speed={r['speed']:.0f} m/s",
        axis=1,
    )

    if allowed is not None:
        df = df[df[["nodes", "channels"]].apply(tuple, axis=1).isin(allowed)]

    metrics = [
        ("pdr", "PDR", "%", "%.1f%%", "C0"),
        ("avg_delay_s", "Average delay", "s", "%.2f s", "C2"),
        ("energy_per_node", "Average energy per node", "J", "%.2f J", "C3"),
        ("avg_sf", "Average SF", "", "%.1f", "C4"),
    ]

    for metric, name, unit, fmt, color in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        yerr = df[std_col] if std_col in df.columns else None
        fig_width = max(16, 0.6 * len(df))
        fig, ax = plt.subplots(figsize=(fig_width, 8), constrained_layout=True)
        label = f"{name} ({unit})"
        bars = ax.bar(
            range(len(df)),
            df[mean_col],
            yerr=yerr,
            capsize=4,
            color=color,
            label=label,
        )
        ax.set_xlabel("")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(
            df["scenario_label"], rotation=45, ha="right"
        )
        ax.set_ylabel(label)
        if hasattr(ax, "tick_params"):
            ax.tick_params(axis="both", labelsize=16)

        if metric == "pdr":
            cap = 100.0
            ax.set_ylim(0, cap)
            ax.axhline(cap, linestyle="--", color="grey")
        elif metric == "avg_delay_s":
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
            bbox_to_anchor=(0.5, 1.3),
            ncol=1,
            title="N: number of nodes, C: number of channels, speed: m/s",
            framealpha=1.0,
            facecolor="white",
        )
        for ext in formats:
            fig.savefig(
                out_dir / f"{metric}_vs_scenario.{ext}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to aggregated mobility_multichannel.csv")
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
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "jpg", "svg", "eps"),
        help="File formats for output figures",
    )
    parser.add_argument(
        "--allowed",
        nargs="+",
        metavar="N,C",
        default=None,
        help="Allowed node-channel pairs (e.g. 50,1 200,3); show all if omitted",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Scenario names to include (e.g. n50_c1_static n200_c6_static); "
            "show all if omitted"
        ),
    )
    args = parser.parse_args(argv)
    allowed = None
    if args.allowed is not None:
        allowed = set()
        for combo in args.allowed:
            try:
                n, c = combo.split(",")
                allowed.add((int(n), int(c)))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid N,C pair: '{combo}'"
                )
    plot(
        args.csv,
        args.output_dir,
        args.max_delay,
        args.max_energy,
        tuple(args.formats),
        allowed,
        set(args.scenarios) if args.scenarios is not None else None,
    )


if __name__ == "__main__":
    main()
