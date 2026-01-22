"""Utilitaires de traçage pour les figures de l'article C."""

from __future__ import annotations

import csv
import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt

from article_c.common.plotting_style import (
    LEGEND_STYLE,
    SAVEFIG_STYLE,
    apply_plot_style,
    set_network_size_ticks,
)
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
MIXRA_FALLBACK_COLUMNS = ("mixra_opt_fallback", "mixra_fallback", "fallback")
LOGGER = logging.getLogger(__name__)


def place_legend(ax: plt.Axes) -> None:
    """Place la légende en haut du graphique selon les consignes."""
    ax.legend(**LEGEND_STYLE)


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
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, **SAVEFIG_STYLE)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")
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


def _normalize_algo(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_mixra_opt(row: dict[str, object]) -> bool:
    return _normalize_algo(row.get("algo")) == "mixra_opt"


def _mixra_opt_fallback(row: dict[str, object]) -> bool:
    for key in MIXRA_FALLBACK_COLUMNS:
        if key in row:
            return _to_bool(row.get(key))
    return False


def filter_mixra_opt_fallback(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    has_mixra_opt = any(_is_mixra_opt(row) for row in rows)
    filtered = [
        row
        for row in rows
        if not (_is_mixra_opt(row) and _mixra_opt_fallback(row))
    ]
    has_valid_mixra_opt = any(_is_mixra_opt(row) for row in filtered)
    if has_mixra_opt and not has_valid_mixra_opt:
        warnings.warn("MixRA-Opt absent (fallback)", stacklevel=2)
    return filtered


def load_step1_aggregated(
    path: Path,
    *,
    allow_sample: bool = False,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV vide pour Step1: {path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        network_size_value = row.get("density")
        if "network_size" in row and row.get("network_size") not in (None, ""):
            network_size_value = row.get("network_size")
        network_size = _to_float(network_size_value)
        parsed_row: dict[str, object] = {
            "network_size": network_size,
            "algo": row.get("algo", ""),
            "snir_mode": row.get("snir_mode", ""),
            "cluster": row.get("cluster", "all"),
            "mixra_opt_fallback": _to_bool(row.get("mixra_opt_fallback")),
        }
        if "density" in row:
            parsed_row["density"] = _to_float(row.get("density"))
        for key, value in row.items():
            if key in {
                "density",
                "network_size",
                "algo",
                "snir_mode",
                "cluster",
                "mixra_opt_fallback",
            }:
                continue
            parsed_row[key] = _to_float(value)
        parsed.append(parsed_row)
    return parsed


def load_step2_aggregated(
    path: Path,
    *,
    allow_sample: bool = False,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV vide pour Step2: {path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        network_size_value = row.get("density")
        if "network_size" in row and row.get("network_size") not in (None, ""):
            network_size_value = row.get("network_size")
        network_size = _to_float(network_size_value)
        parsed_row: dict[str, object] = {
            "network_size": network_size,
            "algo": row.get("algo", ""),
            "snir_mode": row.get("snir_mode", ""),
            "cluster": row.get("cluster", "all"),
        }
        if "density" in row:
            parsed_row["density"] = _to_float(row.get("density"))
        for key, value in row.items():
            if key in {
                "density",
                "network_size",
                "algo",
                "snir_mode",
                "cluster",
            }:
                continue
            parsed_row[key] = _to_float(value)
        parsed.append(parsed_row)
    return parsed


def load_step2_selection_probs(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV vide pour Step2 (sélections): {path}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {
            "round": int(_to_float(row.get("round"))),
            "sf": int(_to_float(row.get("sf"))),
            "selection_prob": _to_float(row.get("selection_prob")),
        }
        if "network_size" in row and row.get("network_size") not in (None, ""):
            parsed_row["network_size"] = _to_float(row.get("network_size"))
        parsed.append(parsed_row)
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
    canonical = _normalize_algo(algo)
    return ALGO_LABELS.get(canonical, algo)


def filter_cluster(rows: list[dict[str, object]], cluster: str) -> list[dict[str, object]]:
    if any("cluster" in row for row in rows):
        return [row for row in rows if row.get("cluster") == cluster]
    return rows


def ensure_network_size(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row.get("network_size") in (None, "") and "density" in row:
            row["network_size"] = row["density"]


def normalize_network_size_rows(rows: list[dict[str, object]]) -> None:
    ensure_network_size(rows)
    for row in rows:
        row["network_size"] = int(_to_float(row.get("network_size")))


def warn_if_missing_network_sizes(
    requested: Iterable[int] | None,
    available: Iterable[int],
) -> None:
    if not requested:
        return
    requested_sizes = sorted({int(_to_float(size)) for size in requested})
    available_sizes = sorted({int(_to_float(size)) for size in available})
    missing = sorted(set(requested_sizes) - set(available_sizes))
    if missing:
        warnings.warn(
            "Tailles de réseau absentes: "
            + ", ".join(str(size) for size in missing)
            + ". Tailles disponibles: "
            + ", ".join(str(size) for size in available_sizes),
            stacklevel=2,
        )


def _network_size_value(row: dict[str, object]) -> int:
    if "network_size" in row:
        return int(_to_float(row.get("network_size")))
    return int(_to_float(row.get("density")))


def filter_rows_by_network_sizes(
    rows: list[dict[str, object]],
    network_sizes: Iterable[int] | None,
) -> tuple[list[dict[str, object]], list[int]]:
    normalize_network_size_rows(rows)
    unique_network_sizes = sorted(
        {row["network_size"] for row in rows if "network_size" in row}
    )
    LOGGER.info("network_size uniques après conversion: %s", unique_network_sizes)
    available = sorted({_network_size_value(row) for row in rows})
    if not network_sizes:
        return rows, available
    requested = sorted({int(_to_float(size)) for size in network_sizes})
    warn_if_missing_network_sizes(requested, available)
    filtered = [row for row in rows if row["network_size"] in requested]
    if not filtered:
        warnings.warn(
            "Aucune taille de réseau trouvée. Tailles disponibles: "
            + ", ".join(str(size) for size in available),
            stacklevel=2,
        )
    return filtered, available


def plot_metric_by_snir(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
) -> None:
    network_sizes = sorted({_network_size_value(row) for row in rows})
    median_key, lower_key, upper_key = resolve_percentile_keys(rows, metric_key)

    def _algo_key(row: dict[str, object]) -> tuple[str, bool]:
        algo_value = str(row.get("algo", ""))
        fallback = bool(row.get("mixra_opt_fallback")) if _is_mixra_opt(row) else False
        return algo_value, fallback

    algorithms = sorted({_algo_key(row) for row in rows})
    for algo, fallback in algorithms:
        algo_rows = [row for row in rows if _algo_key(row) == (algo, fallback)]
        densities = sorted({_network_size_value(row) for row in algo_rows})
        for snir_mode in SNIR_MODES:
            points = {
                _network_size_value(row): row.get(median_key)
                for row in algo_rows
                if row["snir_mode"] == snir_mode
            }
            if not points:
                continue
            values = [
                _value_or_nan(points.get(density, float("nan"))) for density in densities
            ]
            label = f"{algo_label(algo, fallback)} ({SNIR_LABELS[snir_mode]})"
            line = ax.plot(
                densities,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
            )[0]
            if lower_key and upper_key:
                lower_points = {
                    _network_size_value(row): row.get(lower_key)
                    for row in algo_rows
                    if row["snir_mode"] == snir_mode
                }
                upper_points = {
                    _network_size_value(row): row.get(upper_key)
                    for row in algo_rows
                    if row["snir_mode"] == snir_mode
                }
                lower_values = [
                    _value_or_nan(lower_points.get(density, float("nan")))
                    for density in densities
                ]
                upper_values = [
                    _value_or_nan(upper_points.get(density, float("nan")))
                    for density in densities
                ]
                color = line.get_color()
                ax.plot(
                    densities,
                    lower_values,
                    linestyle=":",
                    color=color,
                    alpha=0.6,
                )
                ax.plot(
                    densities,
                    upper_values,
                    linestyle=":",
                    color=color,
                    alpha=0.6,
                )
    set_network_size_ticks(ax, network_sizes)


def resolve_percentile_keys(
    rows: list[dict[str, object]],
    metric_key: str,
) -> tuple[str, str | None, str | None]:
    median_key = metric_key
    lower_key = None
    upper_key = None
    if metric_key.endswith("_mean"):
        base_key = metric_key[: -len("_mean")]
        p10_key = f"{base_key}_p10"
        p50_key = f"{base_key}_p50"
        p90_key = f"{base_key}_p90"
        if any(p50_key in row for row in rows):
            median_key = p50_key
        if any(p10_key in row for row in rows) and any(p90_key in row for row in rows):
            lower_key = p10_key
            upper_key = p90_key
    return median_key, lower_key, upper_key


def plot_metric_by_algo(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int],
    *,
    label_fn: Callable[[object], str] | None = None,
) -> None:
    median_key, lower_key, upper_key = resolve_percentile_keys(rows, metric_key)
    algorithms = sorted({row.get("algo") for row in rows})
    single_size = len(network_sizes) == 1
    only_size = network_sizes[0] if single_size else None
    label_fn = label_fn or (lambda algo: algo_label(str(algo)))
    for algo in algorithms:
        points = {
            int(row["network_size"]): row.get(median_key)
            for row in rows
            if row.get("algo") == algo
        }
        if single_size:
            value = points.get(only_size)
            if _is_invalid_value(value):
                continue
            if lower_key and upper_key:
                low = _value_or_nan(
                    next(
                        (
                            row.get(lower_key)
                            for row in rows
                            if row.get("algo") == algo
                            and int(row["network_size"]) == only_size
                        ),
                        float("nan"),
                    )
                )
                high = _value_or_nan(
                    next(
                        (
                            row.get(upper_key)
                            for row in rows
                            if row.get("algo") == algo
                            and int(row["network_size"]) == only_size
                        ),
                        float("nan"),
                    )
                )
                if not _is_invalid_value(low) and not _is_invalid_value(high):
                    yerr = [[value - low], [high - value]]
                    ax.errorbar([only_size], [value], yerr=yerr, fmt="o", label=label_fn(algo))
                    continue
            ax.scatter([only_size], [value], label=label_fn(algo))
            continue
        values = [_value_or_nan(points.get(size, float("nan"))) for size in network_sizes]
        line = ax.plot(network_sizes, values, marker="o", label=label_fn(algo))[0]
        if lower_key and upper_key:
            lower_points = {
                int(row["network_size"]): row.get(lower_key)
                for row in rows
                if row.get("algo") == algo
            }
            upper_points = {
                int(row["network_size"]): row.get(upper_key)
                for row in rows
                if row.get("algo") == algo
            }
            lower_values = [
                _value_or_nan(lower_points.get(size, float("nan")))
                for size in network_sizes
            ]
            upper_values = [
                _value_or_nan(upper_points.get(size, float("nan")))
                for size in network_sizes
            ]
            color = line.get_color()
            ax.plot(network_sizes, lower_values, linestyle=":", color=color, alpha=0.6)
            ax.plot(network_sizes, upper_values, linestyle=":", color=color, alpha=0.6)


def _is_invalid_value(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _value_or_nan(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


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
                    pdr_std = 0.01 + 0.005 * idx + 0.002 * algos.index(algo)
                    pdr_ci95 = 1.96 * pdr_std / 5**0.5
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
                        "density": density,  # Alias legacy de network_size.
                        "network_size": density,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": cluster,
                        "mixra_opt_fallback": False,
                        "pdr_mean": pdr,
                        "pdr_std": pdr_std,
                        "pdr_ci95": pdr_ci95,
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
                            "density": density,  # Alias legacy de network_size.
                            "network_size": density,
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
