"""Génère des heatmaps UCB1 pour récompense, SNIR et DER (SNIR on/off)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout, save_figure
from plot_defaults import resolve_ieee_figsize
from experiments.ucb1.plots.plot_style import apply_plot_style

DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots"
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace des heatmaps UCB1 (récompense, SNIR, DER) séparées par SNIR on/off."
    )
    parser.add_argument(
        "--ucb1-csv",
        type=Path,
        default=DEFAULT_UCB1,
        help="CSV UCB1 (run_ucb1_load_sweep.py ou export équivalent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les PNG.",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(missing)}")


def _parse_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _detect_snir(row: pd.Series) -> str:
    candidates = [row.get("with_snir"), row.get("use_snir"), row.get("snir_enabled"), row.get("snir")]
    for candidate in candidates:
        parsed = _parse_bool(candidate)
        if parsed is not None:
            return "snir_on" if parsed else "snir_off"
    state = row.get("snir_state")
    if isinstance(state, str) and state:
        state_lower = state.lower()
        if "on" in state_lower:
            return "snir_on"
        if "off" in state_lower:
            return "snir_off"
    return "snir_unknown"


def _resolve_metrics(df: pd.DataFrame) -> dict[str, str]:
    reward_col = "reward_window_mean" if "reward_window_mean" in df.columns else "reward_mean"
    snir_col = "snir_window_mean" if "snir_window_mean" in df.columns else "snir_avg"
    der_col = "der_window" if "der_window" in df.columns else "der"
    return {
        "reward": reward_col,
        "snir": snir_col,
        "der": der_col,
    }


def _build_matrix(
    df: pd.DataFrame,
    metric: str,
    clusters: list[int],
    intervals: list[float],
) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index="cluster", columns="packet_interval_s", values=metric, aggfunc="mean")
        .reindex(index=clusters, columns=intervals)
        .astype(float)
    )
    return pivot


def _format_intervals(intervals: list[float]) -> list[str]:
    labels = []
    for interval in intervals:
        minutes = interval / 60.0
        labels.append(f"{minutes:.1f} min")
    return labels


def _plot_heatmaps(
    df: pd.DataFrame,
    *,
    metrics: dict[str, str],
    output_dir: Path,
    snir_state: str,
    clusters: list[int],
    intervals: list[float],
) -> None:
    titles = {
        "reward": "Récompense",
        "snir": "SNIR",
        "der": "DER",
    }
    labels = {
        "reward": "Récompense",
        "snir": "SNIR (dB)",
        "der": "DER",
    }

    base_width, base_height = resolve_ieee_figsize(max(1, len(clusters)))
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(base_width * 1.5, base_height),
    )
    interval_labels = _format_intervals(intervals)

    for axis, key in zip(axes, ["reward", "snir", "der"], strict=True):
        metric = metrics[key]
        matrix = _build_matrix(df, metric, clusters, intervals)
        image = axis.imshow(matrix.values, aspect="auto", cmap="viridis")
        axis.set_title(titles[key])
        axis.set_xlabel("Intervalle de paquets")
        axis.set_ylabel("Cluster")
        axis.set_xticks(range(len(intervals)))
        axis.set_xticklabels(interval_labels, rotation=45, ha="right")
        axis.set_yticks(range(len(clusters)))
        axis.set_yticklabels([str(cluster) for cluster in clusters])
        fig.colorbar(image, ax=axis, label=labels[key])

    fig.suptitle(f"UCB1 – {SNIR_LABELS.get(snir_state, snir_state)}")
    apply_figure_layout(fig, tight_layout={"rect": [0, 0, 1, 0.93]})

    stem = f"ucb1_reward_snir_der_heatmap_{snir_state}"
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def run_plots(*, csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    metrics = _resolve_metrics(df)
    _ensure_columns(df, ["cluster", "packet_interval_s", *metrics.values()], csv_path)

    df = df.copy()
    df["packet_interval_s"] = pd.to_numeric(df["packet_interval_s"], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["packet_interval_s", "cluster"])
    if df.empty:
        raise ValueError("Aucune donnée disponible après filtrage.")

    df["cluster"] = df["cluster"].astype(int)
    df["snir_state"] = df.apply(_detect_snir, axis=1)

    clusters = sorted(df["cluster"].unique().tolist())
    intervals = sorted(df["packet_interval_s"].unique().tolist())

    for snir_state in ("snir_off", "snir_on"):
        subset = df[df["snir_state"] == snir_state]
        if subset.empty:
            raise ValueError(
                f"Aucune donnée avec SNIR {'activé' if snir_state == 'snir_on' else 'désactivé'}"
            )
        _plot_heatmaps(
            subset,
            metrics=metrics,
            output_dir=output_dir,
            snir_state=snir_state,
            clusters=clusters,
            intervals=intervals,
        )


def main() -> None:
    apply_plot_style()
    args = parse_args()
    run_plots(csv_path=args.ucb1_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
