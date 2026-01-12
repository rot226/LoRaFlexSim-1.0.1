"""Compare la récompense et le regret cumulé pour SNIR on/off."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from experiments.ucb1.plots.plot_style import apply_ieee_style

DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_reward_regret_snir_comparison.png"
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparaison reward/regret SNIR on/off.")
    parser.add_argument("--ucb1-csv", type=Path, default=DEFAULT_UCB1, help="CSV UCB1 (run_ucb1_load_sweep.py).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Chemin du PNG à générer.")
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


def _resolve_reward_col(df: pd.DataFrame) -> str:
    for candidate in ("reward_window_mean", "reward_mean", "reward"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Impossible de trouver une colonne de récompense compatible.")


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    if not args.ucb1_csv.exists():
        raise FileNotFoundError(f"CSV introuvable: {args.ucb1_csv}")

    df = pd.read_csv(args.ucb1_csv)
    reward_col = _resolve_reward_col(df)
    _ensure_columns(df, [reward_col], args.ucb1_csv)

    df = df.copy()
    df["time_s"] = _resolve_time(df)
    df["snir_state"] = df.apply(_detect_snir, axis=1)
    df = df[df["snir_state"].isin(["snir_on", "snir_off"])].dropna(subset=["time_s", reward_col])
    if df.empty:
        raise ValueError("Aucune donnée SNIR on/off disponible pour la comparaison.")

    reward_by_time = (
        df.groupby(["snir_state", "time_s"], as_index=False)[reward_col]
        .mean()
        .sort_values("time_s")
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.0, 6.2), sharex=True)
    for snir_state in ("snir_off", "snir_on"):
        subset = reward_by_time[reward_by_time["snir_state"] == snir_state]
        if subset.empty:
            continue
        axes[0].plot(
            subset["time_s"],
            subset[reward_col],
            label=SNIR_LABELS[snir_state],
            color=SNIR_COLORS[snir_state],
            marker="o",
        )
    axes[0].set_title("Récompense (SNIR)")
    axes[0].set_ylabel("Récompense")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    if axes[0].get_legend_handles_labels()[1]:
        axes[0].legend(ncol=2)

    for snir_state in ("snir_off", "snir_on"):
        subset = reward_by_time[reward_by_time["snir_state"] == snir_state].copy()
        if subset.empty:
            continue
        best_reward = subset[reward_col].max()
        subset["regret"] = best_reward - subset[reward_col]
        subset["cumulative_regret"] = subset["regret"].cumsum()
        axes[1].plot(
            subset["time_s"],
            subset["cumulative_regret"],
            label=SNIR_LABELS[snir_state],
            color=SNIR_COLORS[snir_state],
            marker="o",
        )
    axes[1].set_title("Regret cumulé (SNIR)")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_ylabel("Regret cumulé")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Comparaison SNIR on/off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
