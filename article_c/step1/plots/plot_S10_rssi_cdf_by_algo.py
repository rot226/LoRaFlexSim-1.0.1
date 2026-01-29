"""Trace la figure S10 (CDF RSSI/SNR par algorithme, SNIR on/off)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    algo_label,
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    is_constant_metric,
    render_constant_metric,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize
from article_c.step1.plots.plot_utils import configure_figure

DEFAULT_METRIC_COLUMNS = {
    "rssi": ("rssi_dbm", "rssi_dBm", "rssi", "rssi_db"),
    "snr": ("snr_db", "snr_dB", "snr", "snr_dbm"),
}
TARGET_ALGOS = ("adr", "mixra_h", "mixra_opt")
MIXRA_FALLBACK_COLUMNS = ("mixra_opt_fallback", "mixra_fallback", "fallback")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def _normalize_algo(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "adr": "adr",
        "mixra_h": "mixra_h",
        "mixra_hybrid": "mixra_h",
        "mixra_opt": "mixra_opt",
        "mixra_optimal": "mixra_opt",
        "mixraopt": "mixra_opt",
    }
    return aliases.get(normalized)


def _normalize_snir(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"snir_on", "on", "true", "1", "yes"}:
        return "snir_on"
    if lowered in {"snir_off", "off", "false", "0", "no"}:
        return "snir_off"
    if "on" in lowered:
        return "snir_on"
    if "off" in lowered:
        return "snir_off"
    return None


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_bool(value: str | None) -> bool:
    if value is None or value == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "vrai"}


def _compute_cdf(values: Iterable[float]) -> tuple[list[float], list[float]]:
    sorted_values = sorted(values)
    total = len(sorted_values)
    if total == 0:
        return [], []
    xs = sorted_values
    ys = [(idx + 1) / total for idx in range(total)]
    return xs, ys


def _resolve_metric(
    columns: Iterable[str],
    metric: str,
) -> tuple[str, str, str]:
    if metric == "auto":
        for candidate in ("rssi", "snr"):
            column = _pick_column(columns, DEFAULT_METRIC_COLUMNS[candidate])
            if column:
                label = "RSSI (dBm)" if candidate == "rssi" else "SNR (dB)"
                return candidate, column, label
        raise ValueError("Aucune colonne RSSI/SNR trouvée dans le CSV.")

    if metric not in DEFAULT_METRIC_COLUMNS:
        raise ValueError("La métrique doit être 'rssi', 'snr' ou 'auto'.")
    column = _pick_column(columns, DEFAULT_METRIC_COLUMNS[metric])
    if not column:
        raise ValueError(f"Aucune colonne compatible avec {metric} trouvée.")
    label = "RSSI (dBm)" if metric == "rssi" else "SNR (dB)"
    return metric, column, label


def plot_cdf_by_algo(
    rows: list[dict[str, str]],
    metric: str,
    output_dir: Path,
) -> None:
    if not rows:
        raise ValueError("Aucune ligne trouvée dans le CSV.")

    columns = rows[0].keys()
    algo_col = _pick_column(columns, ("algo", "algorithm", "method"))
    snir_col = _pick_column(columns, ("snir_mode", "snir_state", "snir", "with_snir"))
    fallback_col = _pick_column(columns, MIXRA_FALLBACK_COLUMNS)
    if not algo_col or not snir_col:
        raise ValueError("Colonnes 'algo' et 'snir_mode' requises dans le CSV.")

    metric_key, metric_col, metric_label = _resolve_metric(columns, metric)

    values_by_group: dict[tuple[str, bool, str], list[float]] = {}
    for row in rows:
        algo = _normalize_algo(row.get(algo_col))
        if algo not in TARGET_ALGOS:
            continue
        snir_mode = _normalize_snir(row.get(snir_col))
        if snir_mode not in SNIR_LABELS:
            continue
        fallback = _as_bool(row.get(fallback_col)) if fallback_col else False
        if algo != "mixra_opt":
            fallback = False
        if algo == "mixra_opt" and fallback:
            continue
        value = _as_float(row.get(metric_col))
        if value is None:
            continue
        values_by_group.setdefault((algo, fallback, snir_mode), []).append(value)

    if not values_by_group:
        raise ValueError("Aucune donnée RSSI/SNR compatible avec ADR/MixRA trouvée.")

    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(values_by_group)))
    all_values = [value for values in values_by_group.values() for value in values]
    if is_constant_metric(all_values):
        render_constant_metric(fig, ax, legend_handles=None)
        configure_figure(
            fig,
            [ax],
            f"CDF {metric_key.upper()} par algorithme (SNIR on/off)",
            legend_loc="above",
        )
        save_figure(fig, output_dir, "plot_S10")
        plt.close(fig)
        return
    algo_colors = {
        "adr": "#1f77b4",
        "mixra_h": "#ff7f0e",
        "mixra_opt": "#2ca02c",
    }

    algo_keys: list[tuple[str, bool]] = []
    for algo in TARGET_ALGOS:
        for fallback in (False, True):
            if any(key[0] == algo and key[1] == fallback for key in values_by_group):
                algo_keys.append((algo, fallback))

    for algo, fallback in algo_keys:
        for snir_mode in ("snir_on", "snir_off"):
            values = values_by_group.get((algo, fallback, snir_mode), [])
            if not values:
                continue
            xs, ys = _compute_cdf(values)
            label = f"{algo_label(algo, fallback)} ({SNIR_LABELS[snir_mode]})"
            ax.step(
                xs,
                ys,
                where="post",
                label=label,
                color=algo_colors.get(algo),
                linestyle=SNIR_LINESTYLES[snir_mode],
            )

    ax.set_xlabel(metric_label)
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle=":", alpha=0.6)
    configure_figure(
        fig,
        [ax],
        f"CDF {metric_key.upper()} par algorithme (SNIR on/off)",
        legend_loc="above",
    )

    save_figure(fig, output_dir, "plot_S10")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace la CDF RSSI/SNR pour ADR/MixRA (SNIR on/off).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "raw_packets.csv",
        help="Chemin du fichier raw_packets.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=("auto", "rssi", "snr"),
        default="auto",
        help="Métrique à tracer (auto, rssi ou snr).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plots" / "output",
        help="Répertoire de sortie pour la figure.",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    rows = _read_rows(args.input)
    ensure_network_size(rows)
    for row in rows:
        network_size = row.get("network_size")
        if not network_size:
            network_size = row.get("density", "0")
        row["network_size"] = network_size
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    plot_cdf_by_algo(rows, args.metric, args.output_dir)


if __name__ == "__main__":
    main()
