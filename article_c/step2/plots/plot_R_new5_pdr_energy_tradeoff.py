"""Trace le compromis PDR/énergie selon la taille du réseau (Step2)."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    filter_cluster,
    filter_mixra_opt_fallback,
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
ENERGY_CANDIDATES = (
    "energy_per_packet_mean",
    "energy_per_delivered_packet_mean",
    "energy_per_success_mean",
)
ALGOS = ("ucb1_sf", "adr", "mixra_h", "mixra_opt")


def _normalize_algo(algo: object) -> str:
    return str(algo or "").strip().lower().replace("-", "_").replace(" ", "_")


def _select_metric(df: pd.DataFrame, candidates: tuple[str, ...], metric_name: str) -> str:
    for metric in candidates:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().any():
            return metric
    raise ValueError(f"Aucune métrique {metric_name} trouvée parmi: {', '.join(candidates)}")


def _tx_power_w() -> float:
    tx_power_dbm = float(DEFAULT_CONFIG.radio.tx_power_dbm)
    return 10 ** ((tx_power_dbm - 30.0) / 10.0)


def _extract_energy_series_j(df: pd.DataFrame) -> pd.Series:
    for key in ENERGY_CANDIDATES:
        if key not in df.columns:
            continue
        values = pd.to_numeric(df[key], errors="coerce")
        if not values.notna().any():
            continue
        if key == "energy_per_success_mean":
            return values * _tx_power_w()
        return values

    sent = pd.to_numeric(df.get("sent_mean"), errors="coerce")
    delivered = pd.to_numeric(df.get("received_mean"), errors="coerce")
    toa_s = pd.to_numeric(df.get("mean_toa_s_mean"), errors="coerce")
    if sent.notna().any() and delivered.notna().any() and toa_s.notna().any():
        total_energy_j = sent * toa_s * _tx_power_w()
        out = pd.Series(float("nan"), index=df.index, dtype=float)
        valid = delivered > 0.0
        out.loc[valid] = total_energy_j.loc[valid] / delivered.loc[valid]
        return out

    raise ValueError(
        "Impossible de calculer l'énergie/packet. Colonnes attendues: "
        "energy_per_packet_mean / energy_per_delivered_packet_mean / energy_per_success_mean "
        "ou (sent_mean, received_mean, mean_toa_s_mean)."
    )


def _prepare_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "network_size" not in df.columns and "density" in df.columns:
        df["network_size"] = pd.to_numeric(df["density"], errors="coerce")

    pdr_metric = _select_metric(df, PDR_CANDIDATES, "PDR")

    df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce")
    df[pdr_metric] = pd.to_numeric(df[pdr_metric], errors="coerce")
    df["algo_norm"] = df["algo"].map(_normalize_algo)

    if "snir_mode" not in df.columns:
        df["snir_mode"] = "snir_on"

    df = df[df["algo_norm"].isin(ALGOS) & df["snir_mode"].isin(SNIR_MODES)].copy()

    if df.empty:
        return df

    df["energy_per_packet_j"] = _extract_energy_series_j(df)
    df = df.dropna(subset=["network_size", pdr_metric, "energy_per_packet_j"])

    if df.empty:
        return df

    grouped = (
        df.groupby(["algo_norm", "snir_mode", "network_size"], as_index=False)[
            [pdr_metric, "energy_per_packet_j"]
        ]
        .mean()
        .sort_values(["algo_norm", "snir_mode", "network_size"])
    )
    grouped = grouped.rename(columns={pdr_metric: "pdr"})
    return grouped


def _series_label(algo: str, snir_mode: str, metric: str) -> str:
    return f"{algo_label(algo)} · {SNIR_LABELS.get(snir_mode, snir_mode)} · {metric}"


def _plot(df: pd.DataFrame) -> plt.Figure:
    fig, ax_pdr = plt.subplots(figsize=resolve_ieee_figsize(len(ALGOS)))
    ax_energy = ax_pdr.twinx()

    for algo in ALGOS:
        algo_rows = df[df["algo_norm"] == algo]
        if algo_rows.empty:
            continue

        color = ALGO_COLORS.get(algo, "#4c4c4c")
        marker = ALGO_MARKERS.get(algo, "o")

        for snir_mode in SNIR_MODES:
            subset = algo_rows[algo_rows["snir_mode"] == snir_mode]
            if subset.empty:
                continue

            linestyle = SNIR_LINESTYLES.get(snir_mode, "-")
            x_values = subset["network_size"].tolist()

            ax_pdr.plot(
                x_values,
                subset["pdr"].tolist(),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6,
                label=_series_label(algo, snir_mode, "PDR"),
            )
            ax_energy.plot(
                x_values,
                subset["energy_per_packet_j"].tolist(),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                alpha=0.55,
                label=_series_label(algo, snir_mode, "Energy"),
            )

    ax_pdr.set_xlabel("Network size (nodes)")
    ax_pdr.set_ylabel("PDR")
    ax_pdr.set_ylim(0.0, 1.0)
    ax_pdr.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    ax_energy.set_ylabel("Energy per delivered packet (J/packet)")
    ax_pdr.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax_pdr.grid(True, linestyle=":", alpha=0.35)

    handles_pdr, labels_pdr = ax_pdr.get_legend_handles_labels()
    handles_energy, labels_energy = ax_energy.get_legend_handles_labels()
    fig.legend(
        handles_pdr + handles_energy,
        labels_pdr + labels_energy,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"

    rows = load_step2_aggregated(results_path, allow_sample=True)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    rows = filter_mixra_opt_fallback(rows)
    rows = filter_cluster(rows, "all")

    df = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn("Aucune donnée exploitable pour le compromis PDR/énergie.", stacklevel=2)
        return

    fig = _plot(df)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_R_new5_pdr_energy_tradeoff", use_tight=False)
    assert_legend_present(fig, "plot_R_new5_pdr_energy_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    main()
