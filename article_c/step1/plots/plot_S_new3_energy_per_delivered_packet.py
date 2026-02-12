"""Trace l'énergie par paquet délivré (global + clusters si lisible).

Unité de sortie: J/packet ou mJ/packet selon l'échelle des valeurs.
"""

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
    filter_mixra_opt_fallback,
    load_step1_aggregated,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

NETWORK_SIZES = [80, 160, 320, 640, 1280]
ALGOS = ["adr", "mixra_h", "mixra_opt"]
MAX_CLUSTER_PANELS = 4


def _normalize_algo(algo: object) -> str:
    return str(algo).strip().lower().replace("-", "_").replace(" ", "_")


def _tx_power_w() -> float:
    tx_power_dbm = float(DEFAULT_CONFIG.radio.tx_power_dbm)
    return 10 ** ((tx_power_dbm - 30.0) / 10.0)


def _compute_energy_per_delivered_packet(df: pd.DataFrame) -> pd.Series:
    sent = pd.to_numeric(df.get("sent_mean"), errors="coerce")
    delivered = pd.to_numeric(df.get("received_mean"), errors="coerce")
    toa_s = pd.to_numeric(df.get("mean_toa_s_mean"), errors="coerce")
    power_w = _tx_power_w()
    # Énergie radio émise approximée sur le groupe = sent_mean * mean_toa_s_mean * Ptx.
    total_energy_j = sent * toa_s * power_w
    valid = (delivered > 0.0) & delivered.notna() & total_energy_j.notna()
    energy = pd.Series(float("nan"), index=df.index, dtype=float)
    energy.loc[valid] = total_energy_j.loc[valid] / delivered.loc[valid]
    return energy


def _prepare_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "network_size" not in df.columns and "density" in df.columns:
        raise ValueError("Le champ network_size est requis pour ce plot dédié.")

    df = df[df["network_size"].isin(NETWORK_SIZES)].copy()
    df["algo_norm"] = df["algo"].map(_normalize_algo)
    df = df[df["algo_norm"].isin(ALGOS)].copy()

    available_snir = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    if available_snir:
        df = df[df["snir_mode"].isin(available_snir)].copy()
    else:
        df["snir_mode"] = "snir_on"

    required = {"sent_mean", "received_mean", "mean_toa_s_mean"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Impossible de calculer energy/delivered packet. Colonnes manquantes: "
            + ", ".join(missing)
        )

    df["energy_per_delivered_packet_j"] = _compute_energy_per_delivered_packet(df)
    grouped = (
        df.groupby(["cluster", "algo_norm", "snir_mode", "network_size"], as_index=False)[
            "energy_per_delivered_packet_j"
        ]
        .mean()
        .sort_values(["cluster", "snir_mode", "algo_norm", "network_size"])
    )
    return grouped


def _unit_scale(values: pd.Series) -> tuple[float, str]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return 1.0, "J/packet"
    median = float(valid.median())
    # Normalisation stable: on choisit mJ/packet pour les petites valeurs
    # (lecture plus robuste que des décimales très fines en J/packet).
    if median < 0.01:
        return 1e-3, "mJ/packet"
    return 1.0, "J/packet"


def _plot_for_cluster(df: pd.DataFrame, cluster: str) -> plt.Figure:
    subset = df[df["cluster"] == cluster].copy()
    snir_modes_present = [mode for mode in SNIR_MODES if mode in set(subset["snir_mode"])]
    if not snir_modes_present:
        snir_modes_present = ["snir_on"]

    fig, axes = plt.subplots(
        1,
        len(snir_modes_present),
        figsize=resolve_ieee_figsize(len(snir_modes_present)),
        sharey=True,
    )
    if len(snir_modes_present) == 1:
        axes = [axes]

    scale, unit = _unit_scale(subset["energy_per_delivered_packet_j"])

    for ax, snir_mode in zip(axes, snir_modes_present, strict=False):
        subset_snir = subset[subset["snir_mode"] == snir_mode]

        for algo in ALGOS:
            subset_algo = subset_snir[subset_snir["algo_norm"] == algo]
            points = {
                int(row.network_size): float(row.energy_per_delivered_packet_j)
                for row in subset_algo.itertuples(index=False)
            }
            if not points:
                continue
            y_values = [points.get(size, float("nan")) / scale for size in NETWORK_SIZES]
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

        ax.set_title(SNIR_LABELS.get(snir_mode, snir_mode))
        ax.set_xlabel("Network size (nodes)")
        ax.set_xticks(NETWORK_SIZES)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.grid(True, linestyle=":", alpha=0.35)

    axes[0].set_ylabel(f"Energy per delivered packet ({unit})")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(axes) > 1:
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.78, top=0.84, bottom=0.2)
    else:
        axes[0].legend(loc="best", frameon=True)
        fig.subplots_adjust(top=0.84, bottom=0.2)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"

    try:
        rows = load_step1_aggregated(results_path, allow_sample=True)
    except (ValueError, FileNotFoundError):
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return

    rows = filter_mixra_opt_fallback(rows)
    df = _prepare_dataframe(rows)
    if df.empty:
        warnings.warn(
            "Aucune donnée exploitable pour calculer l'énergie par paquet délivré.",
            stacklevel=2,
        )
        return

    output_dir = step_dir / "plots" / "output"

    # Global (cluster all): obligatoire
    if (df["cluster"] == "all").any():
        fig_global = _plot_for_cluster(df, "all")
        save_figure(fig_global, output_dir, "plot_S_new3_energy_per_delivered_packet", use_tight=False)
        assert_legend_present(fig_global, "plot_S_new3_energy_per_delivered_packet")
        plt.close(fig_global)
    else:
        warnings.warn("Cluster 'all' absent: figure globale non générée.", stacklevel=2)

    # Clusters QoS: optionnel si lisible
    clusters = sorted(cluster for cluster in set(df["cluster"]) if cluster != "all")
    if not clusters:
        return
    if len(clusters) > MAX_CLUSTER_PANELS:
        warnings.warn(
            f"{len(clusters)} clusters détectés; génération cluster ignorée (lisibilité).",
            stacklevel=2,
        )
        return

    for cluster in clusters:
        fig_cluster = _plot_for_cluster(df, cluster)
        save_figure(
            fig_cluster,
            output_dir,
            f"plot_S_new3_energy_per_delivered_packet_cluster_{cluster}",
            use_tight=False,
        )
        assert_legend_present(
            fig_cluster,
            f"plot_S_new3_energy_per_delivered_packet_cluster_{cluster}",
        )
        plt.close(fig_cluster)


if __name__ == "__main__":
    main()
