"""Trace l'énergie par paquet délivré (Step2) pour ADR, MixRA-H, MixRA-Opt et UCB1-SF.

Le script vérifie la cohérence d'unité et protège la lisibilité de la légende.
"""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    filter_cluster,
    filter_mixra_opt_fallback,
    load_step2_aggregated,
    save_figure,
    warn_if_inconsistent,
)
from article_c.common.plot_data_source import load_aggregated_rows_for_source
from plot_defaults import resolve_ieee_figsize

ALGOS = ["adr", "mixra_h", "mixra_opt", "ucb1_sf"]
ENERGY_CANDIDATES = (
    "energy_per_packet_mean",
    "energy_per_delivered_packet_mean",
    "energy_per_success_mean",
)


def _normalize_algo(algo: object) -> str:
    return str(algo or "").strip().lower().replace("-", "_").replace(" ", "_")


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
            # En Step2, cette métrique est souvent un airtime/succès (s/packet).
            # Conversion en énergie via Ptx pour garantir la cohérence d'unité.
            return values * _tx_power_w()
        return values

    sent = pd.to_numeric(df.get("sent_mean"), errors="coerce")
    delivered = pd.to_numeric(df.get("received_mean"), errors="coerce")
    toa_s = pd.to_numeric(df.get("mean_toa_s_mean"), errors="coerce")
    if sent.notna().any() and delivered.notna().any() and toa_s.notna().any():
        power_w = _tx_power_w()
        total_energy_j = sent * toa_s * power_w
        energy = pd.Series(float("nan"), index=df.index, dtype=float)
        valid = delivered > 0.0
        energy.loc[valid] = total_energy_j.loc[valid] / delivered.loc[valid]
        return energy

    raise ValueError(
        "Impossible de calculer l'énergie par paquet. "
        "Colonnes attendues: energy_per_packet_mean / energy_per_delivered_packet_mean "
        "/ energy_per_success_mean ou (sent_mean, received_mean, mean_toa_s_mean)."
    )


def _pick_display_unit(values_j: pd.Series) -> tuple[float, str]:
    clean = pd.to_numeric(values_j, errors="coerce").dropna()
    if clean.empty:
        return 1.0, "J/packet"
    median_j = float(clean.median())
    if median_j < 1e-6:
        return 1e-6, "µJ/packet"
    if median_j < 1e-3:
        return 1e-3, "mJ/packet"
    return 1.0, "J/packet"


def _prepare_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "network_size" not in df.columns and "density" in df.columns:
        df["network_size"] = pd.to_numeric(df["density"], errors="coerce")
    df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce")

    df["algo_norm"] = df["algo"].map(_normalize_algo)
    df = df[df["algo_norm"].isin(ALGOS)].copy()

    if "snir_mode" in df.columns and (df["snir_mode"] == "snir_on").any():
        df = df[df["snir_mode"] == "snir_on"].copy()

    df["energy_per_packet_j"] = _extract_energy_series_j(df)
    df = df.dropna(subset=["network_size", "energy_per_packet_j"])

    return (
        df.groupby(["algo_norm", "network_size"], as_index=False)["energy_per_packet_j"]
        .mean()
        .sort_values(["algo_norm", "network_size"])
    )


def _plot(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(ALGOS)))

    scale, unit = _pick_display_unit(df["energy_per_packet_j"])

    for algo in ALGOS:
        subset = df[df["algo_norm"] == algo]
        if subset.empty:
            continue
        x_values = subset["network_size"].tolist()
        y_values = [float(v) / scale for v in subset["energy_per_packet_j"].tolist()]

        ax.plot(
            x_values,
            y_values,
            label=algo_label(algo),
            color=ALGO_COLORS.get(algo, "#4c4c4c"),
            marker=ALGO_MARKERS.get(algo, "o"),
            linewidth=2.0,
            markersize=6,
        )
        warn_if_inconsistent({"x": x_values, "y": y_values, "label": f"{algo_label(algo)} energy"})

    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel(f"Energy per delivered packet ({unit})")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.grid(True, linestyle=":", alpha=0.35)

    legend = ax.legend(loc="best", frameon=True)
    if legend is not None and len(legend.get_texts()) > 4:
        warnings.warn("Légende dense détectée; vérifiez la lisibilité à l'export.", stacklevel=2)
    return fig


def main(source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    rows = load_aggregated_rows_for_source(
        step_dir=step_dir,
        source=LAST_EFFECTIVE_SOURCE,
        step_label="Step2",
        loader=load_step2_aggregated,
        allow_sample=True,
    )
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    rows = filter_mixra_opt_fallback(rows)
    rows = filter_cluster(rows, "all")

    df = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn("Aucune donnée exploitable pour l'énergie par paquet.", stacklevel=2)
        return

    fig = _plot(df)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_R_new2_energy_per_packet", use_tight=False)
    assert_legend_present(fig, "plot_R_new2_energy_per_packet")
    plt.close(fig)


if __name__ == "__main__":
    main()
