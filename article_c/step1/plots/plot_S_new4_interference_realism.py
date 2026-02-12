"""Trace des métriques de réalisme d'interférence à partir de raw_packets.csv.

Métriques tracées (robustes, dérivées des données disponibles):
- Fraction de paires co-SF avec écart RSSI <= 3 dB.
- Nombre moyen d'interférents co-SF par paquet.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
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
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

NETWORK_SIZES = [80, 160, 320, 640, 1280]
RSSI_NEIGHBOR_DB = 3.0


@dataclass(frozen=True)
class GroupMetrics:
    network_size: int
    algo: str
    snir_mode: str
    frac_neighbors_pm3db: float
    mean_co_sf_interferers: float


def _read_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_algo(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_snir_mode(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"on", "snir_on"}:
        return "snir_on"
    if normalized in {"off", "snir_off"}:
        return "snir_off"
    return normalized


def _parse_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _compute_group_metrics(rows: list[dict[str, str]]) -> list[GroupMetrics]:
    grouped: dict[tuple[int, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        packet_id = row.get("packet_id")
        sf_selected = _parse_int(row.get("sf_selected"))
        rssi_dbm = _parse_float(row.get("rssi_dbm"))
        network_size = _parse_int(row.get("network_size"))
        algo = _normalize_algo(row.get("algo"))
        snir_mode = _normalize_snir_mode(row.get("snir_mode")) or "snir_on"
        replication = str(row.get("replication", ""))

        if packet_id in (None, "") or sf_selected is None or rssi_dbm is None:
            continue
        if network_size is None:
            continue
        if network_size not in NETWORK_SIZES:
            continue
        if not algo:
            continue
        if snir_mode not in SNIR_MODES:
            continue
        grouped[(network_size, algo, snir_mode, replication, str(packet_id))].append(row)

    # Agrège d'abord par run (network_size, algo, snir, replication), puis moyenne.
    run_values: dict[tuple[int, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (network_size, algo, snir_mode, replication, _packet_id), packet_rows in grouped.items():
        # Pour un paquet donné, on construit les groupes par SF et calcule:
        # 1) fraction paires co-SF avec |ΔRSSI|<=3dB
        # 2) co-SF interferers moyens par transmission
        sf_to_rssi: dict[int, list[float]] = defaultdict(list)
        for row in packet_rows:
            sf = _parse_int(row.get("sf_selected"))
            rssi = _parse_float(row.get("rssi_dbm"))
            if sf is None or rssi is None:
                continue
            sf_to_rssi[sf].append(rssi)

        pair_close = 0
        pair_total = 0
        co_sf_interferers_sum = 0.0
        tx_count = 0
        for rssi_values in sf_to_rssi.values():
            n = len(rssi_values)
            if n <= 0:
                continue
            tx_count += n
            co_sf_interferers_sum += float(n * (n - 1))
            if n >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        pair_total += 1
                        if abs(rssi_values[i] - rssi_values[j]) <= RSSI_NEIGHBOR_DB:
                            pair_close += 1

        frac_neighbors = float(pair_close) / float(pair_total) if pair_total > 0 else float("nan")
        mean_interferers = co_sf_interferers_sum / float(tx_count) if tx_count > 0 else float("nan")
        run_values[(network_size, algo, snir_mode, replication)].append((frac_neighbors, mean_interferers))

    aggregated: list[GroupMetrics] = []
    by_group: dict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (network_size, algo, snir_mode, _replication), values in run_values.items():
        valid = [(a, b) for (a, b) in values if pd.notna(a) and pd.notna(b)]
        if not valid:
            continue
        frac_mean = sum(item[0] for item in valid) / len(valid)
        inter_mean = sum(item[1] for item in valid) / len(valid)
        by_group[(network_size, algo, snir_mode)].append((frac_mean, inter_mean))

    for (network_size, algo, snir_mode), values in sorted(by_group.items()):
        frac_mean = sum(item[0] for item in values) / len(values)
        inter_mean = sum(item[1] for item in values) / len(values)
        aggregated.append(
            GroupMetrics(
                network_size=network_size,
                algo=algo,
                snir_mode=snir_mode,
                frac_neighbors_pm3db=frac_mean,
                mean_co_sf_interferers=inter_mean,
            )
        )
    return aggregated


def _plot_metrics(metrics: list[GroupMetrics]) -> plt.Figure:
    df = pd.DataFrame([metric.__dict__ for metric in metrics])
    snir_modes = [mode for mode in SNIR_MODES if mode in set(df["snir_mode"])]
    if not snir_modes:
        snir_modes = ["snir_on"]

    fig, axes = plt.subplots(
        2,
        len(snir_modes),
        figsize=resolve_ieee_figsize(len(snir_modes), scale=1.35),
        sharex=True,
        squeeze=False,
    )

    for col, snir_mode in enumerate(snir_modes):
        subset_snir = df[df["snir_mode"] == snir_mode]
        algos = sorted(set(subset_snir["algo"]))

        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for algo in algos:
            subset_algo = subset_snir[subset_snir["algo"] == algo]
            points_frac = {
                int(row.network_size): float(row.frac_neighbors_pm3db)
                for row in subset_algo.itertuples(index=False)
            }
            points_inter = {
                int(row.network_size): float(row.mean_co_sf_interferers)
                for row in subset_algo.itertuples(index=False)
            }
            y_frac = [points_frac.get(size, float("nan")) for size in NETWORK_SIZES]
            y_inter = [points_inter.get(size, float("nan")) for size in NETWORK_SIZES]

            style = {
                "label": algo_label(algo),
                "color": ALGO_COLORS.get(algo, "#4c4c4c"),
                "marker": ALGO_MARKERS.get(algo, "o"),
                "linestyle": SNIR_LINESTYLES.get(snir_mode, "solid"),
                "linewidth": 2.0,
                "markersize": 6,
            }
            ax_top.plot(NETWORK_SIZES, y_frac, **style)
            ax_bottom.plot(NETWORK_SIZES, y_inter, **style)

        ax_top.set_title(SNIR_LABELS.get(snir_mode, snir_mode))
        ax_top.set_ylabel("Fraction voisins ±3 dB")
        ax_top.set_ylim(0.0, 1.0)
        ax_top.grid(True, linestyle=":", alpha=0.35)

        ax_bottom.set_ylabel("Interférents co-SF moyens")
        ax_bottom.set_xlabel("Network size (nodes)")
        ax_bottom.set_xticks(NETWORK_SIZES)
        ax_bottom.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax_bottom.grid(True, linestyle=":", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(snir_modes) > 1:
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.8, top=0.9, bottom=0.12)
    else:
        axes[0, 0].legend(loc="best", frameon=True)
        fig.subplots_adjust(top=0.9, bottom=0.12)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    raw_results_path = step_dir / "results" / "raw_packets.csv"

    raw_rows = _read_raw_rows(raw_results_path)
    if not raw_rows:
        warnings.warn("raw_packets.csv manquant ou vide, figure ignorée.", stacklevel=2)
        return

    metrics = _compute_group_metrics(raw_rows)
    if not metrics:
        warnings.warn(
            "Données insuffisantes pour calculer les métriques de réalisme d'interférence.",
            stacklevel=2,
        )
        return

    fig = _plot_metrics(metrics)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S_new4_interference_realism", use_tight=False)
    assert_legend_present(fig, "plot_S_new4_interference_realism")
    plt.close(fig)


if __name__ == "__main__":
    main()
