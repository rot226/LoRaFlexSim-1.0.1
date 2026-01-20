"""Utilitaires de traçage pour les figures de l'article C."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings
import csv

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from article_c.common.plotting_style import PLOT_STYLE
from article_c.common.utils import ensure_dir
from article_c.common.config import DEFAULT_CONFIG

ALGO_LABELS = {
    "adr": "ADR",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
    "ucb1_sf": "UCB1-SF",
}
SNIR_MODES = ("snir_on", "snir_off")
SNIR_LABELS = {
    "snir_on": "SNIR on",
    "snir_off": "SNIR off",
}
SNIR_LINESTYLES = {
    "snir_on": "solid",
    "snir_off": "dashed",
}


def apply_plot_style() -> None:
    """Applique le style de tracé commun."""
    plt.rcParams.update(PLOT_STYLE)


def place_legend(ax: plt.Axes) -> None:
    """Place la légende en haut du graphique selon les consignes."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    plt.subplots_adjust(top=0.80)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    use_tight: bool = False,
) -> None:
    """Sauvegarde la figure en PNG et PDF dans le répertoire cible.

    Sur Windows, privilégier bbox_inches=None (valeur par défaut).
    """
    ensure_dir(output_dir)
    if use_tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches=None)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "vrai"}


def load_step1_aggregated(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        return _sample_step1_rows()
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {
            "density": _to_float(row.get("density")),
            "algo": row.get("algo", ""),
            "snir_mode": row.get("snir_mode", ""),
            "cluster": row.get("cluster", "all"),
            "mixra_opt_fallback": _to_bool(row.get("mixra_opt_fallback")),
        }
        for key, value in row.items():
            if key in {"density", "algo", "snir_mode", "cluster", "mixra_opt_fallback"}:
                continue
            parsed_row[key] = _to_float(value)
        parsed.append(parsed_row)
    return parsed


def load_step2_aggregated(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        return _sample_step2_rows()
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "density": _to_float(row.get("density")),
                "algo": row.get("algo", ""),
                "snir_mode": row.get("snir_mode", ""),
                "cluster": row.get("cluster", "all"),
                "success_rate_mean": _to_float(row.get("success_rate_mean")),
                "bitrate_norm_mean": _to_float(row.get("bitrate_norm_mean")),
                "energy_norm_mean": _to_float(row.get("energy_norm_mean")),
                "reward_mean": _to_float(row.get("reward_mean")),
            }
        )
    return parsed


def load_step2_selection_probs(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        return _sample_selection_probs()
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "round": int(_to_float(row.get("round"))),
                "sf": int(_to_float(row.get("sf"))),
                "selection_prob": _to_float(row.get("selection_prob")),
            }
        )
    return parsed


def algo_labels(algorithms: Iterable[object]) -> list[str]:
    labels: list[str] = []
    for algo in algorithms:
        if isinstance(algo, tuple) and len(algo) == 2:
            label = algo_label(str(algo[0]), bool(algo[1]))
        else:
            label = algo_label(str(algo))
        labels.append(label)
    return labels


def algo_label(algo: str, fallback: bool = False) -> str:
    if algo == "mixra_opt" and fallback:
        return "MixRA-Opt (fallback)"
    return ALGO_LABELS.get(algo, algo)


def filter_cluster(rows: list[dict[str, object]], cluster: str) -> list[dict[str, object]]:
    if any("cluster" in row for row in rows):
        return [row for row in rows if row.get("cluster") == cluster]
    return rows


def ensure_network_size(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if "network_size" not in row and "density" in row:
            row["network_size"] = row["density"]


def _network_size_value(row: dict[str, object]) -> int:
    if "network_size" in row:
        return int(_to_float(row.get("network_size")))
    return int(_to_float(row.get("density")))


def filter_rows_by_network_sizes(
    rows: list[dict[str, object]],
    network_sizes: Iterable[int] | None,
) -> tuple[list[dict[str, object]], list[int]]:
    ensure_network_size(rows)
    available = sorted({_network_size_value(row) for row in rows})
    if not network_sizes:
        return rows, available
    requested = sorted({int(size) for size in network_sizes})
    missing = sorted(set(requested) - set(available))
    if missing:
        warnings.warn(
            "Tailles de réseau demandées absentes: "
            + ", ".join(str(size) for size in missing),
            stacklevel=2,
        )
    filtered = [row for row in rows if _network_size_value(row) in requested]
    return filtered, available


def plot_metric_by_snir(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
) -> None:
    all_densities = sorted({_network_size_value(row) for row in rows})

    def _algo_key(row: dict[str, object]) -> tuple[str, bool]:
        algo_value = str(row.get("algo", ""))
        fallback = bool(row.get("mixra_opt_fallback")) if algo_value == "mixra_opt" else False
        return algo_value, fallback

    algorithms = sorted({_algo_key(row) for row in rows})
    for algo, fallback in algorithms:
        algo_rows = [row for row in rows if _algo_key(row) == (algo, fallback)]
        densities = sorted({_network_size_value(row) for row in algo_rows})
        for snir_mode in SNIR_MODES:
            points = {
                _network_size_value(row): row[metric_key]
                for row in algo_rows
                if row["snir_mode"] == snir_mode
            }
            if not points:
                continue
            values = [points.get(density, float("nan")) for density in densities]
            label = f"{algo_label(algo, fallback)} ({SNIR_LABELS[snir_mode]})"
            ax.plot(
                densities,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
            )
    ax.set_xticks(all_densities)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))


def _sample_step1_rows() -> list[dict[str, object]]:
    densities = [50, 100, 150]
    algos = ["adr", "mixra_h", "mixra_opt"]
    clusters = list(DEFAULT_CONFIG.qos.clusters) + ["all"]
    sf_values = list(DEFAULT_CONFIG.radio.spreading_factors)
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for algo in algos:
            for idx, density in enumerate(densities):
                for cluster in clusters:
                    base = 0.9 - 0.1 * idx
                    penalty = 0.05 if snir_mode == "snir_off" else 0.0
                    modifier = 0.02 * algos.index(algo)
                    cluster_bonus = 0.02 if cluster == "gold" else 0.0
                    pdr = max(
                        0.0, min(1.0, base - penalty + modifier + cluster_bonus)
                    )
                    algo_idx = algos.index(algo)
                    weights = []
                    for sf in sf_values:
                        sf_idx = sf_values.index(sf)
                        bias = 1.2 - 0.15 * sf_idx + 0.05 * algo_idx
                        if snir_mode == "snir_off":
                            bias += 0.05 * sf_idx
                        weights.append(max(0.05, bias))
                    total_weight = sum(weights) or 1.0
                    sf_shares = [weight / total_weight for weight in weights]
                    base_toa = 40.0 + 8.0 * idx + 4.0 * algo_idx
                    if snir_mode == "snir_off":
                        base_toa += 5.0
                    mean_toa_s = (base_toa + 0.2 * density) / 1000.0
                    row = {
                        "density": density,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": cluster,
                        "mixra_opt_fallback": False,
                        "pdr_mean": pdr,
                        "sent_mean": 120 * density,
                        "received_mean": 120 * density * pdr,
                        "mean_toa_s": mean_toa_s,
                    }
                    for sf, share in zip(sf_values, sf_shares, strict=False):
                        row[f"sf{sf}_share_mean"] = share
                    rows.append(row)
    return rows


def _sample_step2_rows() -> list[dict[str, object]]:
    densities = [50, 100, 150]
    algos = ["ADR", "MixRA-H", "MixRA-Opt", "UCB1-SF"]
    clusters = list(DEFAULT_CONFIG.qos.clusters) + ["all"]
    rows: list[dict[str, object]] = []
    for snir_mode in SNIR_MODES:
        for algo_idx, algo in enumerate(algos):
            for density in densities:
                for cluster in clusters:
                    reward = max(0.2, 0.7 - 0.05 * algo_idx - 0.1 * (density - 0.5))
                    penalty = 0.05 if snir_mode == "snir_off" else 0.0
                    cluster_bonus = 0.03 if cluster == "gold" else 0.0
                    rows.append(
                        {
                            "density": density,
                            "algo": algo,
                            "snir_mode": snir_mode,
                            "cluster": cluster,
                            "success_rate_mean": max(
                                0.3, 0.9 - 0.05 * algo_idx - penalty + cluster_bonus
                            ),
                            "bitrate_norm_mean": 0.4 + 0.1 * algo_idx - penalty,
                            "energy_norm_mean": 0.3 + 0.1 * algo_idx + penalty,
                            "reward_mean": reward - penalty + cluster_bonus,
                        }
                    )
    return rows


def _sample_selection_probs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_id in range(1, 11):
        for sf in (7, 8, 9, 10, 11, 12):
            rows.append(
                {
                    "round": round_id,
                    "sf": sf,
                    "selection_prob": max(0.05, 0.25 - 0.01 * (sf - 7) + 0.01 * round_id),
                }
            )
    return rows
