"""Plot packet delivery ratio, delay and energy metrics for the load sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from scripts.mne3sd.common import apply_ieee_style, prepare_figure_directory, save_figure
from plot_defaults import DEFAULT_FIGSIZE_SIMPLE

ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "pdr_load_summary.csv"
ARTICLE = "article_a"
SCENARIO = "pdr_load"
PLOT_METRICS = ("pdr", "delay", "energy")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate figures for the load sweep experiments, highlighting packet "
            "delivery ratio, average delay and per-node energy consumption."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the pdr_load_summary.csv file",
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
    return parser.parse_args(argv)


def load_summary(path: Path) -> pd.DataFrame:
    """Load and normalise the summary CSV used for the plots."""

    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}. Run simulate_pdr_load.py first."
        )

    df = pd.read_csv(path)
    required_columns = {"interval_s", "pdr_mean"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["interval_s"] = pd.to_numeric(df["interval_s"], errors="coerce")
    if df["interval_s"].isna().any():
        raise ValueError("Interval column contains invalid values")

    df.sort_values("interval_s", inplace=True)

    for column in ("mode", "sf_assignment"):
        if column in df.columns:
            df[column] = df[column].astype(str)

    for column in ("adr_node", "adr_server"):
        if column in df.columns:
            df[column] = df[column].astype(int)

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

    mode = mapping.get("mode")
    if mode not in (None, "", "nan"):
        parts.append(str(mode).capitalize())

    sf_assignment = mapping.get("sf_assignment")
    if sf_assignment not in (None, "", "nan"):
        parts.append(str(sf_assignment))

    if "adr_node" in mapping or "adr_server" in mapping:
        node_state = mapping.get("adr_node")
        server_state = mapping.get("adr_server")
        if node_state is not None and server_state is not None:
            parts.append(
                f"ADR node {'on' if int(node_state) else 'off'}, server {'on' if int(server_state) else 'off'}"
            )
        elif node_state is not None:
            parts.append(f"ADR node {'on' if int(node_state) else 'off'}")
        elif server_state is not None:
            parts.append(f"ADR server {'on' if int(server_state) else 'off'}")

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


def plot_pdr(summary: pd.DataFrame, *, figures_dir: Path | None) -> None:
    """Render the packet delivery ratio plot."""

    pdr_column = "pdr_mean"
    if pdr_column not in summary.columns:
        raise ValueError("The summary file does not contain pdr_mean")

    group_columns = [
        column for column in ("mode", "sf_assignment", "adr_node", "adr_server") if column in summary.columns
    ]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SIMPLE)

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("interval_s")
        y = ordered[pdr_column] * 100.0
        yerr = None
        if "pdr_std" in summary.columns:
            yerr = ordered["pdr_std"].fillna(0.0) * 100.0
        ax.errorbar(
            ordered["interval_s"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else None,
            label=label,
        )

    ax.set_xlabel("Reporting interval (s)")
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
        metric="pdr_vs_interval",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_load_pdr_vs_interval", output_dir)


def plot_delay(summary: pd.DataFrame, *, figures_dir: Path | None) -> bool:
    """Render the average delay plot if data is available."""

    if "avg_delay_s_mean" not in summary.columns:
        return False

    group_columns = [
        column for column in ("mode", "sf_assignment", "adr_node", "adr_server") if column in summary.columns
    ]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SIMPLE)

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("interval_s")
        y = ordered["avg_delay_s_mean"]
        err_values = None
        capsize = None
        if "avg_delay_s_std" in summary.columns:
            err_values = ordered["avg_delay_s_std"].fillna(0.0)
            capsize = 3
        ax.errorbar(
            ordered["interval_s"],
            y,
            yerr=err_values,
            marker="o",
            capsize=capsize,
            label=label,
        )

    ax.set_xlabel("Reporting interval (s)")
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
        metric="delay_vs_interval",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_load_delay_vs_interval", output_dir)
    return True


def plot_energy(summary: pd.DataFrame, *, figures_dir: Path | None) -> bool:
    """Render the energy-per-node plot if the data is available."""

    if "energy_per_node_J_mean" not in summary.columns:
        return False

    group_columns = [
        column for column in ("mode", "sf_assignment", "adr_node", "adr_server") if column in summary.columns
    ]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SIMPLE)

    for group_key, group in _iterate_groups(summary, group_columns):
        label = _configuration_label(group_columns, group_key)
        ordered = group.sort_values("interval_s")
        y = ordered["energy_per_node_J_mean"]
        err_values = None
        capsize = None
        if "energy_per_node_J_std" in summary.columns:
            err_values = ordered["energy_per_node_J_std"].fillna(0.0)
            capsize = 3
        ax.errorbar(
            ordered["interval_s"],
            y,
            yerr=err_values,
            marker="o",
            capsize=capsize,
            label=label,
        )

    ax.set_xlabel("Reporting interval (s)")
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
        metric="energy_vs_interval",
        base_dir=figures_dir,
    )
    save_figure(fig, "pdr_load_energy_vs_interval", output_dir)
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    apply_ieee_style()
    if args.style:
        plt.style.use(args.style)

    summary = load_summary(args.results)

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
