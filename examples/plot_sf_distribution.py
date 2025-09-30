"""Plot the distribution of Spreading Factors from CSV files."""
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main(files: list[str], output_dir: Path, basename: str) -> None:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sf_cols = [c for c in df.columns if c.startswith("sf_distribution.")]
    if sf_cols:
        sf_counts = {int(c.split(".")[1]): int(df[c].sum()) for c in sf_cols}
    else:
        sf_cols = [c for c in df.columns if c.startswith("sf")]
        sf_counts = {int(c[2:]): int(df[c].sum()) for c in sf_cols}
    if not sf_counts:
        print("No SF column found in provided files.")
        return
    sfs = sorted(sf_counts)
    counts = [sf_counts[sf] for sf in sfs]
    for sf, count in zip(sfs, counts):
        print(f"SF{sf}: {count}")
    plt.bar(sfs, counts)
    plt.xlabel("Spreading Factor")
    plt.ylabel("Packets")
    plt.grid(True)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        description="Plot SF distribution and save as PNG, JPG and EPS",
    )
    parser.add_argument("files", nargs="+", help="Metrics CSV files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current folder)",
    )
    parser.add_argument(
        "--basename",
        default="sf_distribution",
        help="Base filename without extension",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.files, args.output_dir, args.basename)

