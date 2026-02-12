"""Trace le PDR global Step2 par taille de réseau avec comparaison SNIR."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    apply_plot_style,
    assert_legend_present,
    load_step2_aggregated,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

PDR_CANDIDATES = (
    "pdr_global_mean",
    "pdr_mean",
    "success_rate_mean",
    "success_rate",
)


def _select_pdr_metric(df: pd.DataFrame) -> str:
    for metric in PDR_CANDIDATES:
        if metric in df.columns:
            series = pd.to_numeric(df[metric], errors="coerce")
            if series.notna().any():
                return metric
    raise ValueError(
        "Aucune métrique PDR trouvée. Colonnes attendues: "
        + ", ".join(PDR_CANDIDATES)
    )


def _prepare_dataframe(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ""

    metric_key = _select_pdr_metric(df)

    if "network_size" not in df.columns and "density" in df.columns:
        df["network_size"] = pd.to_numeric(df["density"], errors="coerce")

    if "cluster" in df.columns and (df["cluster"] == "all").any():
        df = df[df["cluster"] == "all"].copy()

    df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce")
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    df = df.dropna(subset=["network_size", metric_key])

    available_snir = [mode for mode in SNIR_MODES if mode in set(df.get("snir_mode", []))]
    if available_snir:
        df = df[df["snir_mode"].isin(available_snir)].copy()
    else:
        df["snir_mode"] = "snir_on"

    grouped = (
        df.groupby(["snir_mode", "network_size"], as_index=False)[metric_key]
        .mean()
        .sort_values(["snir_mode", "network_size"])
    )
    return grouped, metric_key


def _plot(df: pd.DataFrame, metric_key: str) -> plt.Figure:
    snir_modes_present = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(max(1, len(snir_modes_present))))

    for snir_mode in snir_modes_present:
        subset = df[df["snir_mode"] == snir_mode]
        x_values = subset["network_size"].tolist()
        y_values = subset[metric_key].tolist()
        ax.plot(
            x_values,
            y_values,
            label=SNIR_LABELS.get(snir_mode, snir_mode),
            linestyle=SNIR_LINESTYLES.get(snir_mode, "-"),
            marker="o",
            linewidth=2.0,
            markersize=6,
        )

    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel("PDR global")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregates" / "aggregated_results.csv"

    rows = load_step2_aggregated(results_path, allow_sample=True)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    df, metric_key = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn("Aucune donnée exploitable pour tracer le PDR global.", stacklevel=2)
        return

    fig = _plot(df, metric_key)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_R_new1_pdr_global", use_tight=False)
    assert_legend_present(fig, "plot_R_new1_pdr_global")
    plt.close(fig)


if __name__ == "__main__":
    main()
