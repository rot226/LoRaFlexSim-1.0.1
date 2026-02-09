"""Trace un nuage de points récompense moyenne vs PDR (raw_all)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    algo_label,
    apply_figure_layout,
    apply_plot_style,
    filter_rows_by_network_sizes,
    place_legend,
    save_figure,
    warn_metric_checks,
)
from plot_defaults import resolve_ieee_figsize

ALGO_ALIASES = {
    "adr": "adr",
    "ADR": "adr",
    "mixra_h": "mixra_h",
    "MixRA-H": "mixra_h",
    "mixra_opt": "mixra_opt",
    "MixRA-Opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "UCB1-SF": "ucb1_sf",
}
TARGET_ALGOS = ("ucb1_sf", "adr", "mixra_h", "mixra_opt")


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        print(
            "ERREUR: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée."
        )
        return True
    return False


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object, default: int = 0) -> int:
    return int(_to_float(value, default=default))


def _load_step1_raw_results(
    path: Path,
    network_sizes: list[int] | None,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    parsed: list[dict[str, object]] = []
    for row in rows:
        algo = _canonical_algo(str(row.get("algo", "")))
        if algo is None:
            continue
        if row.get("cluster", "") != "all":
            continue
        if row.get("snir_mode") != "snir_on":
            continue
        pdr = _to_float_or_none(row.get("pdr"))
        if pdr is None:
            continue
        network_size = _to_float_or_none(row.get("network_size") or row.get("density"))
        if network_size is None:
            continue
        replication = row.get("replication")
        if replication in (None, ""):
            replication = row.get("seed")
        parsed.append(
            {
                "network_size": int(network_size),
                "algo": algo,
                "replication": _to_int(replication),
                "seed": _to_int(row.get("seed")),
                "pdr": pdr,
                "sent": _to_float(row.get("sent"), default=0.0),
            }
        )
    filtered, _ = filter_rows_by_network_sizes(parsed, network_sizes)
    return filtered


def _load_step2_raw_results(
    path: Path,
    network_sizes: list[int] | None,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    parsed: list[dict[str, object]] = []
    for row in rows:
        algo = _canonical_algo(str(row.get("algo", "")))
        if algo is None:
            continue
        if row.get("cluster", "") != "all":
            continue
        if row.get("snir_mode") != "snir_on":
            continue
        reward = _to_float_or_none(row.get("reward"))
        if reward is None:
            continue
        network_size = _to_float_or_none(row.get("network_size") or row.get("density"))
        if network_size is None:
            continue
        parsed.append(
            {
                "network_size": int(network_size),
                "algo": algo,
                "replication": _to_int(row.get("replication")),
                "reward": reward,
                "traffic_sent": _to_float(row.get("traffic_sent"), default=0.0),
            }
        )
    filtered, _ = filter_rows_by_network_sizes(parsed, network_sizes)
    return filtered


def _collect_points(
    step1_results_path: Path,
    step2_results_path: Path,
    network_sizes: list[int] | None,
) -> list[dict[str, float | int | str]]:
    step1_rows = _load_step1_raw_results(step1_results_path, network_sizes)
    step2_rows = _load_step2_raw_results(step2_results_path, network_sizes)
    pdr_by_key: dict[tuple[int, str, int], dict[str, object]] = {}
    for row in step1_rows:
        key = (int(row["network_size"]), str(row["algo"]), int(row["replication"]))
        pdr_by_key[key] = row
    points: list[dict[str, float | int | str]] = []
    for row in step2_rows:
        key = (int(row["network_size"]), str(row["algo"]), int(row["replication"]))
        if key not in pdr_by_key:
            continue
        network_size, algo, replication = key
        if algo not in TARGET_ALGOS:
            continue
        pdr_row = pdr_by_key[key]
        traffic_value = float(row.get("traffic_sent", 0.0))
        size_value = (
            traffic_value
            if traffic_value > 0.0
            else float(pdr_row.get("sent", 0.0) or network_size)
        )
        points.append(
            {
                "algo": algo,
                "network_size": network_size,
                "replication": replication,
                "reward_mean": float(row["reward"]),
                "pdr": float(pdr_row["pdr"]),
                "size_value": size_value,
            }
        )
    return points


def _scale_sizes(
    values: list[float],
    *,
    min_size: float = 35.0,
    max_size: float = 140.0,
) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [0.5 * (min_size + max_size) for _ in values]
    scaled: list[float] = []
    for value in values:
        ratio = (value - min_value) / (max_value - min_value)
        scaled.append(min_size + ratio * (max_size - min_size))
    return scaled


def _add_size_legend(
    ax: plt.Axes,
    size_values: list[float],
    sizes: list[float],
) -> None:
    if not size_values or not sizes:
        return
    min_value = min(size_values)
    max_value = max(size_values)
    median_value = sorted(size_values)[len(size_values) // 2]
    reference_values = [
        ("min", min_value),
        ("médiane", median_value),
        ("max", max_value),
    ]
    size_map = dict(zip(size_values, sizes, strict=False))
    handles = []
    labels = []
    for label, value in reference_values:
        closest = min(size_values, key=lambda x: abs(x - value))
        handles.append(
            ax.scatter([], [], s=size_map.get(closest, 50.0), color="gray", alpha=0.6)
        )
        labels.append(f"{label}: {closest:.0f}")
    legend = ax.legend(
        handles,
        labels,
        title="Taille (trafic ou nœuds)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(legend)


def _apply_jitter(
    value: float,
    rng: random.Random,
    jitter: float,
    lower: float,
    upper: float,
) -> float:
    jittered = value + rng.uniform(-jitter, jitter)
    return max(lower, min(upper, jittered))


def _plot_scatter(points: list[dict[str, float | int | str]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(TARGET_ALGOS)))
    size_values = [float(point["size_value"]) for point in points]
    pdr_values = [float(point["pdr"]) for point in points]
    reward_values = [float(point["reward_mean"]) for point in points]
    warn_metric_checks(
        reward_values,
        "Récompense moyenne",
    )
    warn_metric_checks(
        sorted(pdr_values),
        "PDR (raw_all)",
        min_value=0.0,
        max_value=1.0,
        expected_monotonic="nondecreasing",
    )
    sizes = _scale_sizes(size_values)
    labeled_algos: set[str] = set()
    rng = random.Random(7)
    for point, size in zip(points, sizes, strict=False):
        algo = str(point["algo"])
        label = algo_label(algo) if algo not in labeled_algos else None
        if label is not None:
            labeled_algos.add(algo)
        jittered_pdr = _apply_jitter(float(point["pdr"]), rng, 0.006, 0.0, 1.05)
        jittered_reward = _apply_jitter(
            float(point["reward_mean"]), rng, 0.006, 0.0, 1.05
        )
        ax.scatter(
            jittered_pdr,
            jittered_reward,
            label=label,
            s=size,
            color=ALGO_COLORS.get(algo, "#4c4c4c"),
            marker=ALGO_MARKERS.get(algo, "o"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
        )
        size_label = f"{float(point['size_value']):.0f}"
        ax.annotate(
            size_label,
            (jittered_pdr, jittered_reward),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.75,
        )
    ax.set_xlabel("PDR (par réplication)")
    ax.set_ylabel("Récompense moyenne (fenêtre)")
    ax.set_title("Step 2 - Récompense vs PDR (raw_all)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    place_legend(ax)
    _add_size_legend(ax, size_values, sizes)
    apply_figure_layout(fig, tight_layout=True)
    return fig


def main() -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args()
    network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    root_dir = Path(__file__).resolve().parent
    step_dir = root_dir / "article_c" / "step2"
    step1_results_path = root_dir / "article_c" / "step1" / "results" / "raw_results.csv"
    step2_results_path = step_dir / "results" / "raw_all.csv"
    points = _collect_points(
        step1_results_path,
        step2_results_path,
        network_sizes,
    )

    fig = _plot_scatter(points)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL10_reward_vs_pdr_scatter", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
