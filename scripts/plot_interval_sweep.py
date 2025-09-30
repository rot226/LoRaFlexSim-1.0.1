"""Plot PDR and collisions versus packet interval.

This script reads ``results/interval_summary.csv`` produced by
``scripts/run_interval_sweep.py``.  It computes the mean ``PDR(%)`` and
``collisions`` for each interval and generates a two-panel figure saved to
``figures/pdr_collisions_vs_interval.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    summary_path = RESULTS_DIR / "interval_summary.csv"
    df = pd.read_csv(summary_path)
    agg = (
        df.groupby("interval")[["PDR(%)", "collisions"]]
        .mean()
        .reset_index()
        .sort_values("interval")
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    axes[0].plot(agg["interval"], agg["PDR(%)"], marker="o")
    axes[0].set_ylabel("PDR (%)")
    axes[0].set_title("PDR vs Interval")

    axes[1].plot(agg["interval"], agg["collisions"], marker="o")
    axes[1].set_ylabel("Collisions")
    axes[1].set_xlabel("Interval")
    axes[1].set_title("Collisions vs Interval")

    fig.tight_layout()
    base = FIGURES_DIR / "pdr_collisions_vs_interval"
    for ext in ("png", "jpg", "eps"):
        dpi = 300 if ext in ("png", "jpg", "eps") else None
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
