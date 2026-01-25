"""Plot packet delivery ratio, delay and energy metrics for the network size (number of nodes) sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from scripts.mne3sd.common import apply_ieee_style, prepare_figure_directory, save_figure

ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "pdr_density_summary.csv"
ARTICLE = "article_a"
SCENARIO = "pdr_density"
PLOT_METRICS = ("pdr", "delay", "energy")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate figures for the network size (number of nodes) sweep, including packet delivery "
            "ratio, mean delay and per-node energy consumption."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the pdr_density_summary.csv file",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        help="Override the root directory where figures are written",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=PLOT_METRICS,
        default=list(PLOT_METRICS),
        help="Subset of metrics to render (default: all)",
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
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filter network sizes (e.g., --network-sizes 100 200 300).",
    )
    return parser.parse_args(argv)


def load_summary(path: Path) -> pd.DataFrame:
    """Load and normalise the summary CSV used for the plots."""

    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}. Run simulate_pdr_density.py first."
        )

    df = pd.read_csv(path)
    required_columns = {"density_gw_per_km2", "nodes", "sf_mode", "pdr_mean"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["density_gw_per_km2"] = pd.to_numeric(df["density_gw_per_km2"], errors="coerce")
    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce")
    if df["density_gw_per_km2"].isna().any() or df["nodes"].isna().any():
        raise ValueError("Gateway density or network size (number of nodes) columns contain invalid values")

    df.sort_values(["sf_mode", "nodes", "density_gw_per_km2"], inplace=True)

    df["sf_mode"] = df["sf_mode"].astype(str)

    numeric_columns = [
        col
        for col in df.columns
        if col.endswith("_mean") or col.endswith("_std") or col in {"pdr_std"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _configuration_label(keys: Iterable[str], values: Sequence[object]) -> str:
    parts: list[str] = []
    mapping = {key: value for key, value in zip(keys, values)}

    nodes = mapping.get("nodes")
    if nodes not in (None, "", "nan"):
        parts.append(f"{int(float(nodes))} nodes")

    sf_mode = mapping.get("sf_mode")
    if sf_mode not in (None, "", "nan"):
        mode_text = str(sf_mode)
        if mode_text.lower() == "adaptive":
            parts.append("Adaptive SF")
        elif mode_text.lower().startswith("fixed"):
            suffix = mode_text[5:]
            parts.append(f"Fixed SF{suffix}")
        else:
            parts.append(mode_text)

    return ", ".join(parts) if parts else "All configurations"


def _iterate_groups(
    df: pd.DataFrame, keys: Sequence[str]
) -> Iterator[Tuple[Tuple[object, ...], pd.DataFrame]]:
    """Yield grouped data keyed by ``keys`` preserving single-key tuples."""

    if keys:
        for group_key, group in df.groupby(list(keys), dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            yield group_key, group
    else:
        yield tuple(), df


def _filter_network_sizes(
    df: pd.DataFrame,
    network_sizes: Sequence[int] | None,
) -> pd.DataFrame:
    if not network_sizes:
        return df
    available = sorted(df["nodes"].dropna().unique())
    requested = sorted({int(size) for size in network_sizes})
    missing = sorted(set(requested) - {int(value) for value in available})
    if missing:
        warnings.warn(
            "Tailles de réseau demandées absentes: "
            + ", ".join(str(size) for size in missing),
            stacklevel=2,
        )
    return df[df["nodes"].isin(requested)]


def plot_pdr(summary: pd.DataFrame, *, figures_dir: Path | None) -> None:
    """Render the packet delivery ratio plot."""

    fig, ax = plt.subplots()
    group_columns = [column for column in ("nodes", "sf_mode") if column in summary.columns]

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("density_gw_per_km2")
        y = ordered["pdr_mean"] * 100.0
        yerr = None
        if "pdr_std" in summary.columns:
            yerr = ordered["pdr_std"].fillna(0.0) * 100.0
        ax.errorbar(
            ordered["density_gw_per_km2"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else None,
            label=label,
        )

    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Packet delivery ratio (%)")
    ax.set_ylim(0, 105)
    fig.legend(
        title="Configuration",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.subplots_adjust(top=0.80)

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="pdr_vs_density",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_density_pdr_vs_density", output_dir)


def plot_delay(summary: pd.DataFrame, *, figures_dir: Path | None) -> bool:
    """Render the average delay plot if data is available."""

    if "avg_delay_s_mean" not in summary.columns:
        return False

    group_columns = [column for column in ("nodes", "sf_mode") if column in summary.columns]
    fig, ax = plt.subplots()

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("density_gw_per_km2")
        y = ordered["avg_delay_s_mean"]
        err_values = None
        capsize = None
        if "avg_delay_s_std" in summary.columns:
            err_values = ordered["avg_delay_s_std"].fillna(0.0)
            capsize = 3
        ax.errorbar(
            ordered["density_gw_per_km2"],
            y,
            yerr=err_values,
            marker="o",
            capsize=capsize,
            label=label,
        )

    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Average delay (s)")
    fig.legend(
        title="Configuration",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.subplots_adjust(top=0.80)

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="delay_vs_density",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_density_delay_vs_density", output_dir)
    return True


def plot_energy(summary: pd.DataFrame, *, figures_dir: Path | None) -> bool:
    """Render the energy-per-node plot if the data is available."""

    if "energy_per_node_J_mean" not in summary.columns:
        return False

    group_columns = [column for column in ("nodes", "sf_mode") if column in summary.columns]
    fig, ax = plt.subplots()

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("density_gw_per_km2")
        y = ordered["energy_per_node_J_mean"]
        err_values = None
        capsize = None
        if "energy_per_node_J_std" in summary.columns:
            err_values = ordered["energy_per_node_J_std"].fillna(0.0)
            capsize = 3
        ax.errorbar(
            ordered["density_gw_per_km2"],
            y,
            yerr=err_values,
            marker="o",
            capsize=capsize,
            label=label,
        )

    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Energy per node (J)")
    fig.legend(
        title="Configuration",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.subplots_adjust(top=0.80)

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_vs_density",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_density_energy_vs_density", output_dir)
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    apply_ieee_style()
    if args.style:
        plt.style.use(args.style)

    summary = _filter_network_sizes(load_summary(args.results), args.network_sizes)

    selected_metrics = list(dict.fromkeys(args.metrics)) or list(PLOT_METRICS)

    if "pdr" in selected_metrics:
        plot_pdr(summary, figures_dir=args.figures_dir)
    if "delay" in selected_metrics:
        created = plot_delay(summary, figures_dir=args.figures_dir)
        if not created:
            print("Average delay data unavailable; skipping delay plot.")
    if "energy" in selected_metrics:
        created = plot_energy(summary, figures_dir=args.figures_dir)
        if not created:
            print("Energy per node data unavailable; skipping energy plot.")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
