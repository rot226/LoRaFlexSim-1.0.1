"""Trace des courbes temporelles MAB (regret cumulé, DER, SNIR, énergie)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout, save_figure
from plot_defaults import resolve_ieee_figsize
from experiments.ucb1.plots.plot_style import apply_plot_style, filter_top_groups, top_groups

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots"
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
LINE_STYLES = ["-", "--", ":", "-."]
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé", "snir_unknown": "SNIR inconnu"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génère les graphiques MAB pour UCB1.")
    parser.add_argument(
        "--ucb1-csv",
        type=Path,
        default=DEFAULT_UCB1,
        help="CSV UCB1 contenant des fenêtres temporelles (run_ucb1_load_sweep.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les PNG.",
    )
    parser.add_argument(
        "--packet-interval",
        type=float,
        action="append",
        default=[],
        help="Filtre les intervalles de paquets (peut être répété).",
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


def _resolve_time(df: pd.DataFrame) -> pd.Series:
    if "window_start_s" in df.columns:
        return pd.to_numeric(df["window_start_s"], errors="coerce")
    if "window_index" in df.columns and "packet_interval_s" in df.columns:
        return pd.to_numeric(df["window_index"], errors="coerce") * pd.to_numeric(
            df["packet_interval_s"], errors="coerce"
        )
    if "window_index" in df.columns:
        return pd.to_numeric(df["window_index"], errors="coerce")
    return pd.Series([0.0 for _ in range(len(df))], index=df.index)


def _style_maps(clusters: list[int], intervals: list[float]) -> tuple[dict[int, str], dict[float, str]]:
    cluster_colors = {cluster: PALETTE[index % len(PALETTE)] for index, cluster in enumerate(clusters)}
    interval_styles = {
        interval: LINE_STYLES[index % len(LINE_STYLES)] for index, interval in enumerate(intervals)
    }
    return cluster_colors, interval_styles


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    apply_figure_layout(fig, tight_layout=True)
    save_figure(fig, output_dir, stem)
    plt.close(fig)
    return output_dir / f"{stem}.png"


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
    clusters: list[int],
    intervals: list[float],
    colors: dict[int, str],
    styles: dict[float, str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(max(1, len(clusters))))
    for cluster in clusters:
        for interval in intervals:
            subset = df[(df["cluster"] == cluster) & (df["packet_interval_s"] == interval)]
            subset = subset.dropna(subset=["time_s", metric])
            if subset.empty:
                continue
            subset = subset.sort_values("time_s")
            label = f"Cluster {cluster} ({interval:.0f}s)"
            ax.plot(
                subset["time_s"],
                subset[metric],
                label=label,
                color=colors[cluster],
                linestyle=styles[interval],
                marker="o",
                markersize=3,
            )
    ax.set_title(title)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)
    _save_plot(fig, output_dir, output_name)


def _plot_metric_snir_overlay(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
    clusters: list[int],
    intervals: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(max(1, len(clusters))))
    for cluster in clusters:
        for interval in intervals:
            for snir_state in ("snir_off", "snir_on"):
                subset = df[
                    (df["cluster"] == cluster)
                    & (df["packet_interval_s"] == interval)
                    & (df["snir_state"] == snir_state)
                ]
                subset = subset.dropna(subset=["time_s", metric])
                if subset.empty:
                    continue
                subset = subset.sort_values("time_s")
                label = f"C{cluster} ({interval:.0f}s, {SNIR_LABELS[snir_state]})"
                ax.plot(
                    subset["time_s"],
                    subset[metric],
                    label=label,
                    color=SNIR_COLORS[snir_state],
                    linestyle=LINE_STYLES[int(intervals.index(interval)) % len(LINE_STYLES)],
                    marker="o",
                    markersize=3,
                )
    ax.set_title(title)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)
    _save_plot(fig, output_dir, output_name)


def run_plots(*, csv_path: Path, output_dir: Path, packet_intervals: list[float]) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    _ensure_columns(
        df,
        [
            "cluster",
            "packet_interval_s",
            "window_index",
            "reward_window_mean",
            "der_window",
            "snir_window_mean",
            "energy_window_mean",
        ],
        csv_path,
    )
    df = df.copy()
    df["time_s"] = _resolve_time(df)
    df["packet_interval_s"] = pd.to_numeric(df["packet_interval_s"], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["time_s", "packet_interval_s", "cluster"])
    if packet_intervals:
        df = df[df["packet_interval_s"].isin(packet_intervals)]
    if df.empty:
        raise ValueError("Aucune donnée disponible après filtrage.")

    df["cluster"] = df["cluster"].astype(int)
    df_limited = filter_top_groups(df, ["cluster", "packet_interval_s"], max_groups=3)
    clusters = sorted(df_limited["cluster"].unique().tolist())
    intervals = sorted(df_limited["packet_interval_s"].unique().tolist())
    colors, styles = _style_maps(clusters, intervals)

    df["snir_state"] = df.apply(_detect_snir, axis=1)
    df_limited["snir_state"] = df_limited.apply(_detect_snir, axis=1)

    top_pair = top_groups(df, ["cluster", "packet_interval_s"], max_groups=1)
    df_snir = (
        df.merge(
            pd.DataFrame(top_pair, columns=["cluster", "packet_interval_s"]),
            on=["cluster", "packet_interval_s"],
            how="inner",
        )
        if top_pair
        else df
    )
    df_snir["snir_state"] = df_snir.apply(_detect_snir, axis=1)

    regret_frames = []
    for cluster in clusters:
        for interval in intervals:
            subset = df_limited[(df_limited["cluster"] == cluster) & (df_limited["packet_interval_s"] == interval)]
            if subset.empty:
                continue
            subset = subset.sort_values("time_s").copy()
            best_reward = subset["reward_window_mean"].max()
            subset["regret"] = best_reward - subset["reward_window_mean"]
            subset["cumulative_regret"] = subset["regret"].cumsum()
            regret_frames.append(subset)
    regret_df = pd.concat(regret_frames, ignore_index=True)
    regret_snir_frames = []
    for cluster in clusters:
        for interval in intervals:
            for snir_state in ("snir_off", "snir_on", "snir_unknown"):
                subset = df_limited[
                    (df_limited["cluster"] == cluster)
                    & (df_limited["packet_interval_s"] == interval)
                    & (df_limited["snir_state"] == snir_state)
                ]
                if subset.empty:
                    continue
                subset = subset.sort_values("time_s").copy()
                best_reward = subset["reward_window_mean"].max()
                subset["regret"] = best_reward - subset["reward_window_mean"]
                subset["cumulative_regret"] = subset["regret"].cumsum()
                regret_snir_frames.append(subset)
    regret_snir_df = (
        pd.concat(regret_snir_frames, ignore_index=True) if regret_snir_frames else regret_df.copy()
    )

    _plot_metric(
        regret_df,
        metric="cumulative_regret",
        title="Regret cumulé",
        ylabel="Regret cumulé",
        output_name="ucb1_mab_cumulative_regret.png",
        clusters=clusters,
        intervals=intervals,
        colors=colors,
        styles=styles,
        output_dir=output_dir,
    )
    _plot_metric_snir_overlay(
        regret_snir_df.merge(
            pd.DataFrame(top_pair, columns=["cluster", "packet_interval_s"]),
            on=["cluster", "packet_interval_s"],
            how="inner",
        )
        if top_pair
        else regret_snir_df,
        metric="cumulative_regret",
        title="Regret cumulé (SNIR)",
        ylabel="Regret cumulé",
        output_name="ucb1_mab_cumulative_regret_snir_overlay.png",
        clusters=sorted(df_snir["cluster"].unique().tolist()),
        intervals=sorted(df_snir["packet_interval_s"].unique().tolist()),
        output_dir=output_dir,
    )
    _plot_metric_snir_overlay(
        df_snir,
        metric="reward_window_mean",
        title="Récompense (SNIR)",
        ylabel="Récompense (fenêtre glissante)",
        output_name="ucb1_mab_reward_vs_time_snir_overlay.png",
        clusters=sorted(df_snir["cluster"].unique().tolist()),
        intervals=sorted(df_snir["packet_interval_s"].unique().tolist()),
        output_dir=output_dir,
    )
    _plot_metric(
        df_limited,
        metric="der_window",
        title="DER",
        ylabel="DER",
        output_name="ucb1_mab_der_vs_time.png",
        clusters=clusters,
        intervals=intervals,
        colors=colors,
        styles=styles,
        output_dir=output_dir,
    )
    _plot_metric(
        df_limited,
        metric="snir_window_mean",
        title="SNIR moyen",
        ylabel="SNIR (dB)",
        output_name="ucb1_mab_snir_vs_time.png",
        clusters=clusters,
        intervals=intervals,
        colors=colors,
        styles=styles,
        output_dir=output_dir,
    )
    _plot_metric(
        df_limited,
        metric="energy_window_mean",
        title="Énergie moyenne",
        ylabel="Énergie (J)",
        output_name="ucb1_mab_energy_vs_time.png",
        clusters=clusters,
        intervals=intervals,
        colors=colors,
        styles=styles,
        output_dir=output_dir,
    )


def main() -> None:
    apply_plot_style()
    args = parse_args()
    run_plots(
        csv_path=args.ucb1_csv,
        output_dir=args.output_dir,
        packet_intervals=args.packet_interval,
    )


if __name__ == "__main__":
    main()
