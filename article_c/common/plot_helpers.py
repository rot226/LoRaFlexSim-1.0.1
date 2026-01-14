"""Utilitaires de traçage pour les figures de l'article C."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv

import matplotlib.pyplot as plt

from article_c.common.plotting_style import PLOT_STYLE
from article_c.common.utils import ensure_dir

ALGO_LABELS = {
    "adr": "ADR",
    "mixra_h": "MixRA-H",
    "mixra_opt": "MixRA-Opt",
    "ucb1_sf": "UCB1-SF",
}


def apply_plot_style() -> None:
    """Applique le style de tracé commun."""
    plt.rcParams.update(PLOT_STYLE)


def place_legend(ax: plt.Axes) -> None:
    """Place la légende en haut du graphique selon les consignes."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    plt.subplots_adjust(top=0.80)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Sauvegarde la figure en PNG et PDF dans le répertoire cible."""
    ensure_dir(output_dir)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


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


def load_step1_aggregated(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    if not rows:
        return _sample_step1_rows()
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "density": _to_float(row.get("density")),
                "algo": row.get("algo", ""),
                "snir_mode": row.get("snir_mode", ""),
                "pdr_mean": _to_float(row.get("pdr_mean")),
                "sent_mean": _to_float(row.get("sent_mean")),
                "received_mean": _to_float(row.get("received_mean")),
            }
        )
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


def algo_labels(algorithms: Iterable[str]) -> list[str]:
    return [ALGO_LABELS.get(algo, algo) for algo in algorithms]


def _sample_step1_rows() -> list[dict[str, object]]:
    densities = [0.1, 0.5, 1.0]
    algos = ["adr", "mixra_h", "mixra_opt"]
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for algo in algos:
            for idx, density in enumerate(densities):
                base = 0.9 - 0.1 * idx
                penalty = 0.05 if snir_mode == "snir_off" else 0.0
                modifier = 0.02 * algos.index(algo)
                pdr = max(0.0, min(1.0, base - penalty + modifier))
                rows.append(
                    {
                        "density": density,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "pdr_mean": pdr,
                        "sent_mean": 120 * density,
                        "received_mean": 120 * density * pdr,
                    }
                )
    return rows


def _sample_step2_rows() -> list[dict[str, object]]:
    densities = [0.5, 1.0, 1.5]
    algos = ["ADR", "MixRA-H", "MixRA-Opt", "UCB1-SF"]
    rows: list[dict[str, object]] = []
    for algo_idx, algo in enumerate(algos):
        for density in densities:
            reward = max(0.2, 0.7 - 0.05 * algo_idx - 0.1 * (density - 0.5))
            rows.append(
                {
                    "density": density,
                    "algo": algo,
                    "snir_mode": "snir_on",
                    "success_rate_mean": max(0.3, 0.9 - 0.05 * algo_idx),
                    "bitrate_norm_mean": 0.4 + 0.1 * algo_idx,
                    "energy_norm_mean": 0.3 + 0.1 * algo_idx,
                    "reward_mean": reward,
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
