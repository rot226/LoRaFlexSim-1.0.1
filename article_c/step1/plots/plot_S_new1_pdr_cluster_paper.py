"""Trace un plot dédié PDR (paper) pour APRA/Aimi/MixRA-Opt/MixRA-H avec SNIR on/off."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    filter_mixra_opt_fallback,
    load_step1_aggregated,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

NETWORK_SIZES = [80, 160, 320, 640, 1280]
ALGOS = ["apra", "aimi", "mixra_opt", "mixra_h"]
PDR_CLUSTER_TARGETS = [0.90, 0.80, 0.70]


def _normalize_algo(algo: object) -> str:
    return str(algo).strip().lower().replace("-", "_").replace(" ", "_")


def _prepare_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "network_size" not in df.columns and "density" in df.columns:
        raise ValueError("Le champ network_size est requis pour ce plot dédié.")

    df = df[df["network_size"].isin(NETWORK_SIZES)].copy()
    df["algo_norm"] = df["algo"].map(_normalize_algo)
    df = df[df["algo_norm"].isin(ALGOS)].copy()
    df = df[df["snir_mode"].isin(SNIR_MODES)].copy()

    grouped = (
        df.groupby(["algo_norm", "snir_mode", "network_size"], as_index=False)["pdr_mean"]
        .mean()
        .sort_values(["algo_norm", "snir_mode", "network_size"])
    )
    return grouped


def _plot(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        len(SNIR_MODES),
        figsize=resolve_ieee_figsize(len(SNIR_MODES)),
        sharey=True,
    )
    if len(SNIR_MODES) == 1:
        axes = [axes]

    for ax, snir_mode in zip(axes, SNIR_MODES, strict=False):
        subset_snir = df[df["snir_mode"] == snir_mode]

        for algo in ALGOS:
            subset_algo = subset_snir[subset_snir["algo_norm"] == algo]
            points = {
                int(row.network_size): float(row.pdr_mean)
                for row in subset_algo.itertuples(index=False)
            }
            if not points:
                continue

            y_values = [points.get(size, float("nan")) for size in NETWORK_SIZES]
            ax.plot(
                NETWORK_SIZES,
                y_values,
                label=algo_label(algo),
                color=ALGO_COLORS.get(algo, "#4c4c4c"),
                marker=ALGO_MARKERS.get(algo, "o"),
                linestyle=SNIR_LINESTYLES.get(snir_mode, "solid"),
                linewidth=2.0,
                markersize=6,
            )

        for idx, target in enumerate(PDR_CLUSTER_TARGETS, start=1):
            ax.axhline(
                y=target,
                color="red",
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
                label=f"Cible cluster C{idx} ({target:.2f})",
            )

        snir_label = SNIR_LABELS.get(snir_mode, snir_mode)
        ax.set_xlabel("Network size (nodes)")
        ax.set_xticks(NETWORK_SIZES)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"PDR (prob.) — {snir_label}")
        ax.grid(True, linestyle=":", alpha=0.35)


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True)
    fig.subplots_adjust(right=0.74, bottom=0.2)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"

    rows = load_step1_aggregated(results_path, allow_sample=True)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    rows = filter_mixra_opt_fallback(rows)
    df = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn(
            "Aucune donnée disponible pour les algos/network sizes demandés.",
            stacklevel=2,
        )
        return

    fig = _plot(df)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S_new1_pdr_cluster_paper", use_tight=False)
    assert_legend_present(fig, "plot_S_new1_pdr_cluster_paper")
    plt.close(fig)


if __name__ == "__main__":
    main()
