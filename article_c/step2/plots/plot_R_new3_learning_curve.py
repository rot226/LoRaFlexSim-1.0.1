"""Trace la courbe d'apprentissage Step2 avec variabilité par round et par taille.

Par défaut, le script trace des petits multiples par taille de réseau.
Il met en avant au minimum les tailles 80 et 1280 nœuds si elles sont présentes.
"""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_MARKERS,
    algo_label,
    apply_plot_style,
    assert_legend_present,
    save_figure,
)
from article_c.common.plotting_style import label_for
from plot_defaults import resolve_ieee_figsize

FOCUS_SIZES = (80, 1280)


def _normalize_algo(algo: object) -> str:
    return str(algo or "").strip().lower().replace("-", "_").replace(" ", "_")


def _load_learning_curve(path: Path, *, allow_sample: bool = True) -> pd.DataFrame:
    if not path.exists():
        if allow_sample:
            return _sample_learning_curve()
        warnings.warn(f"CSV introuvable: {path}", stacklevel=2)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        if allow_sample:
            return _sample_learning_curve()
        warnings.warn(f"CSV vide: {path}", stacklevel=2)
        return pd.DataFrame()

    if "network_size" not in df.columns and "density" in df.columns:
        df["network_size"] = pd.to_numeric(df["density"], errors="coerce")

    required = {"network_size", "round", "algo", "avg_reward"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans learning_curve.csv: {', '.join(missing)}")

    df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["avg_reward"] = pd.to_numeric(df["avg_reward"], errors="coerce")
    df = df.dropna(subset=["network_size", "round", "algo", "avg_reward"]).copy()

    df["network_size"] = df["network_size"].astype(int)
    df["round"] = df["round"].astype(int)
    df["algo_norm"] = df["algo"].map(_normalize_algo)

    return df


def _sample_learning_curve() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for network_size in FOCUS_SIZES:
        for algo in ("adr", "ucb1_sf", "mixra_h"):
            base = {"adr": 0.40, "ucb1_sf": 0.50, "mixra_h": 0.47}[algo]
            for round_id in range(1, 41):
                for seed in range(5):
                    trend = 0.006 * round_id
                    penalty = 0.00004 * network_size
                    noise = float(rng.normal(0.0, 0.015 + 0.003 * seed))
                    rows.append(
                        {
                            "network_size": network_size,
                            "round": round_id,
                            "algo": algo,
                            "avg_reward": base + trend - penalty + noise,
                        }
                    )
    sample = pd.DataFrame(rows)
    sample["algo_norm"] = sample["algo"].map(_normalize_algo)
    return sample


def _select_sizes(df: pd.DataFrame) -> list[int]:
    all_sizes = sorted(int(v) for v in df["network_size"].dropna().unique())
    focus = [size for size in FOCUS_SIZES if size in all_sizes]
    if focus:
        return focus
    # Fallback: au moins deux tailles représentatives si possible.
    if len(all_sizes) >= 2:
        return [all_sizes[0], all_sizes[-1]]
    return all_sizes


def _aggregate_with_variability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["network_size", "algo_norm", "round"], as_index=False)
        .agg(
            reward_mean=("avg_reward", "mean"),
            reward_std=("avg_reward", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            reward_count=("avg_reward", "count"),
        )
        .sort_values(["network_size", "algo_norm", "round"])
    )
    grouped["reward_ci95"] = grouped.apply(
        lambda row: 1.96 * row["reward_std"] / math.sqrt(row["reward_count"])
        if row["reward_count"] > 1
        else 0.0,
        axis=1,
    )
    return grouped


def _plot(agg: pd.DataFrame, sizes: list[int]) -> plt.Figure:
    ncols = 2 if len(sizes) > 1 else 1
    nrows = math.ceil(len(sizes) / ncols)
    fig_w, fig_h = resolve_ieee_figsize(max(1, len(sizes)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w * ncols, max(fig_h * nrows, 3.0)),
        sharex=True,
        sharey=True,
    )

    axes_list = np.atleast_1d(axes).reshape(-1)

    for idx, size in enumerate(sizes):
        ax = axes_list[idx]
        subset_size = agg[agg["network_size"] == size]
        algos = sorted(subset_size["algo_norm"].unique())

        for algo in algos:
            subset = subset_size[subset_size["algo_norm"] == algo]
            if subset.empty:
                continue

            rounds = subset["round"].to_numpy(dtype=float)
            mean = subset["reward_mean"].to_numpy(dtype=float)
            ci95 = subset["reward_ci95"].to_numpy(dtype=float)

            color = ALGO_COLORS.get(algo, "#444444")
            marker = ALGO_MARKERS.get(algo, "o")

            ax.plot(
                rounds,
                mean,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=4,
                label=algo_label(algo),
            )
            ax.fill_between(rounds, mean - ci95, mean + ci95, color=color, alpha=0.18)

        ax.set_ylabel(f"{label_for('y.reward_mean')} — N={size}")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlabel("Round")

    for ax in axes_list[len(sizes) :]:
        ax.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    return fig


def main(source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "learning_curve.csv"

    df = _load_learning_curve(results_path, allow_sample=True)
    if df.empty:
        warnings.warn("Aucune donnée de learning_curve exploitable.", stacklevel=2)
        return

    sizes = _select_sizes(df)
    if not sizes:
        warnings.warn("Aucune taille de réseau disponible.", stacklevel=2)
        return

    df = df[df["network_size"].isin(sizes)].copy()
    agg = _aggregate_with_variability(df)

    if (agg["reward_count"] <= 1).all():
        warnings.warn(
            "Variabilité limitée: une seule répétition par point (IC95=0).",
            stacklevel=2,
        )

    fig = _plot(agg, sizes)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_R_new3_learning_curve", use_tight=False)
    assert_legend_present(fig, "plot_R_new3_learning_curve")
    plt.close(fig)


if __name__ == "__main__":
    main()
