#!/usr/bin/env python3
"""Plot the initial positions of simulated nodes."""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loraflexsim.launcher.simulator import Simulator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-nodes", type=int, default=100, help="Number of nodes to simulate"
    )
    parser.add_argument(
        "--area-size", type=float, default=1000.0, help="Side of the square area"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="figures/node_positions.png",
        help="Path to save the scatter plot",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=100.0,
        help="Marker size for node positions",
    )
    args = parser.parse_args(argv)

    sim = Simulator(
        num_nodes=args.num_nodes,
        area_size=args.area_size,
        seed=args.seed,
        mobility=False,
    )
    positions = [(n.x, n.y) for n in sim.nodes]
    xs, ys = zip(*positions)

    gateway_positions = [(g.x, g.y) for g in sim.gateways]
    gx, gy = (zip(*gateway_positions) if gateway_positions else ([], []))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    node_points = ax.scatter(
        xs,
        ys,
        s=args.marker_size,
        edgecolors="black",
        facecolors="C0",
        label="Nodes",
    )
    for n in sim.nodes:
        ax.annotate(
            str(n.id),
            (n.x, n.y),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    legend_handles = [node_points]
    if gateway_positions:
        gateway_points = ax.scatter(
            gx,
            gy,
            marker="*",
            s=200,
            edgecolors="black",
            facecolors="red",
            label="Gateways",
        )
        for g in sim.gateways:
            ax.annotate(
                str(g.id),
                (g.x, g.y),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        legend_handles.append(gateway_points)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    plt.subplots_adjust(top=0.80)
    for ext in ("png", "jpg", "eps"):
        dpi = 300 if ext in ("png", "jpg", "eps") else None
        fig.savefig(
            output_path.with_suffix(f".{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
