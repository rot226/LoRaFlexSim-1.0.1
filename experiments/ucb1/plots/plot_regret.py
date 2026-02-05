"""Superpose le regret cumulé UCB1 en distinguant SNIR activé/désactivé."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout
from experiments.ucb1.plots.plot_style import apply_ieee_style, top_groups

DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_regret_snir_overlay.png"

SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé", "snir_unknown": "SNIR inconnu"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Superpose le regret cumulé UCB1 (SNIR on/off).")
    parser.add_argument("--ucb1-csv", type=Path, default=DEFAULT_UCB1, help="CSV UCB1 avec regret_cumulative.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG de sortie.")
    parser.add_argument(
        "--packet-interval",
        type=float,
        action="append",
        default=[],
        help="Filtre les intervalles de paquets (peut être répété).",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
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


def plot_regret(*, csv_path: Path, output_path: Path, packet_intervals: list[float]) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    _ensure_columns(
        df,
        ["cluster", "packet_interval_s", "window_index", "regret_cumulative"],
        csv_path,
    )
    df = df.copy()
    df["time_s"] = _resolve_time(df)
    df["packet_interval_s"] = pd.to_numeric(df["packet_interval_s"], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df["regret_cumulative"] = pd.to_numeric(df["regret_cumulative"], errors="coerce")
    df = df.dropna(subset=["time_s", "packet_interval_s", "cluster", "regret_cumulative"])
    if packet_intervals:
        df = df[df["packet_interval_s"].isin(packet_intervals)]
    if df.empty:
        raise ValueError("Aucune donnée disponible après filtrage.")

    df["cluster"] = df["cluster"].astype(int)
    df["snir_state"] = df.apply(_detect_snir, axis=1)

    top_pair = top_groups(df, ["cluster", "packet_interval_s"], max_groups=1)
    if top_pair:
        df = df.merge(
            pd.DataFrame(top_pair, columns=["cluster", "packet_interval_s"]),
            on=["cluster", "packet_interval_s"],
            how="inner",
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    clusters = sorted(df["cluster"].unique().tolist())
    intervals = sorted(df["packet_interval_s"].unique().tolist())

    for cluster in clusters:
        for interval in intervals:
            for snir_state in ("snir_off", "snir_on"):
                subset = df[
                    (df["cluster"] == cluster)
                    & (df["packet_interval_s"] == interval)
                    & (df["snir_state"] == snir_state)
                ]
                subset = subset.dropna(subset=["time_s", "regret_cumulative"])
                if subset.empty:
                    continue
                subset = subset.sort_values("time_s")
                label = f"Cluster {cluster} ({interval:.0f}s, {SNIR_LABELS[snir_state]})"
                ax.plot(
                    subset["time_s"],
                    subset["regret_cumulative"],
                    label=label,
                    color=SNIR_COLORS[snir_state],
                    marker="o",
                    markersize=3,
                )

    ax.set_title("Regret cumulé (SNIR)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Regret cumulé")
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_layout(fig, tight_layout=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    plot_regret(
        csv_path=args.ucb1_csv,
        output_path=args.output,
        packet_intervals=args.packet_interval,
    )


if __name__ == "__main__":
    main()
