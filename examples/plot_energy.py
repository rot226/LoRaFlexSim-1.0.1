"""Plot total or per-node energy consumption from CSV files."""
import sys
import argparse
from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from article_c.common.plotting_style import apply_base_rcparams


def main(
    files: list[str],
    per_node: bool,
    output_dir: Path,
    basename: str | None,
    network_sizes: list[int] | None,
) -> None:
    apply_base_rcparams()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if network_sizes and "nodes" in df.columns:
        available = sorted(df["nodes"].dropna().unique())
        requested = sorted({int(size) for size in network_sizes})
        missing = sorted(set(requested) - {int(value) for value in available})
        if missing:
            warnings.warn(
                "Tailles de réseau demandées absentes: "
                + ", ".join(str(size) for size in missing),
                stacklevel=2,
            )
        df = df[df["nodes"].isin(requested)]
    output_dir.mkdir(parents=True, exist_ok=True)
    if per_node:
        cols = [c for c in df.columns if c.startswith("energy_by_node.")]
        if not cols:
            print("No per-node consumption columns found.")
            return
        energy = {int(c.split(".")[1]): df[c].mean() for c in cols}
        nodes = sorted(energy)
        values = [energy[n] for n in nodes]
        plt.bar(nodes, values)
        plt.xlabel("Node")
        plt.ylabel("Energy consumed (J)")
        plt.grid(True)
        if basename is None:
            basename = "energy_per_node"
    else:
        col = "energy_J" if "energy_J" in df.columns else "energy"
        if col not in df.columns:
            print("No energy_J or energy column found.")
            return
        if "nodes" in df.columns:
            series = df.groupby("nodes")[col].mean()
            series.plot(marker="o")
            plt.xlabel("Number of nodes")
        else:
            series = df[col]
            series.plot(marker="o")
            plt.xlabel("Run")
        plt.ylabel("Energy consumed (J)")
        plt.grid(True)
        if basename is None:
            basename = "energy_total"
    plt.subplots_adjust(top=0.80)
    for ext, params in {
        "png": {"dpi": 300},
        "jpg": {"dpi": 300},
        "eps": {"dpi": 300},
    }.items():
        path = output_dir / f"{basename}.{ext}"
        plt.savefig(path, bbox_inches="tight", pad_inches=0, **params)
        print(f"Figure saved to {path}")
    plt.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot energy consumption and save the figure as PNG, JPG and EPS",
    )
    parser.add_argument("files", nargs="+", help="Metrics CSV files")
    parser.add_argument(
        "--per-node",
        action="store_true",
        help="Plot consumption per node",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current folder)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Base filename without extension. Defaults to 'energy_total' or 'energy_per_node' depending on --per-node",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filter network sizes (e.g., --network-sizes 100 200 300).",
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.files, args.per_node, args.output_dir, args.basename, args.network_sizes)
