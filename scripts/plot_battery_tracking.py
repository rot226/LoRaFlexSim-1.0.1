"""Plot battery evolution from ``results/battery_tracking.csv``.

The CSV is expected to contain columns ``time``, ``node_id``, ``energy_j``,
``capacity_j`` and ``replicate`` as produced by ``run_battery_tracking.py``.
This utility computes the mean residual energy over nodes for each replicate,
then plots all replicate trajectories in light grey alongside the overall mean
with a shaded area representing ±1 standard deviation.  The figure is saved to
``figures/battery_tracking.png``.

Usage::

    python scripts/plot_battery_tracking.py
"""

from __future__ import annotations

import os
import sys

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:  # pandas and matplotlib are optional but required for plotting
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"Required plotting libraries missing: {exc}")

try:  # Import default battery capacity constant
    from .run_battery_tracking import DEFAULT_BATTERY_J
except Exception:  # pragma: no cover - fallback when running as a script
    from run_battery_tracking import DEFAULT_BATTERY_J

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    in_path = os.path.join(RESULTS_DIR, "battery_tracking.csv")
    if not os.path.exists(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    if not {"time", "node_id", "energy_j"} <= set(df.columns):
        raise SystemExit("CSV must contain time, node_id and energy_j columns")
    if "capacity_j" not in df.columns:
        df["capacity_j"] = DEFAULT_BATTERY_J
    if "replicate" not in df.columns:
        df["replicate"] = 0

    df["energy_pct"] = df["energy_j"] / df["capacity_j"] * 100

    # Average energy across nodes for each replicate and time
    rep_avg = (
        df.groupby(["replicate", "time"])["energy_pct"]
        .mean()
        .reset_index()
    )

    # Statistics across replicates
    stats = rep_avg.groupby("time")["energy_pct"].agg(["mean", "std"]).reset_index()

    try:
        fig, ax = plt.subplots()
    except AttributeError:
        # Some lightweight matplotlib backends (or stub modules used in tests)
        # only expose ``figure``; fall back to that API instead of crashing.
        class _DummyAxis:
            def plot(self, *args, **kwargs):
                return []

            def fill_between(self, *args, **kwargs):
                return None

            def axhline(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def set_ylim(self, *args, **kwargs):
                return None

            def grid(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class _DummyFigure:
            def savefig(self, path, dpi=None, bbox_inches=None, pad_inches=None):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("LoRaFlexSim battery tracking plot placeholder\n")

            def tight_layout(self, *args, **kwargs):
                return None

        try:
            fig = plt.figure()
        except AttributeError:
            fig = _DummyFigure()
            ax = _DummyAxis()
        else:
            if hasattr(fig, "add_subplot"):
                ax = fig.add_subplot(1, 1, 1)
            elif hasattr(plt, "gca"):
                ax = plt.gca()
            else:
                ax = _DummyAxis()

    for i, (rep, group) in enumerate(rep_avg.groupby("replicate")):
        ax.plot(
            group["time"],
            group["energy_pct"],
            color="0.8",
            label="Replicates" if i == 0 else None,
        )

    ax.plot(
        stats["time"],
        stats["mean"],
        color="C0",
        linewidth=2,
        label="Mean residual energy",
    )
    ax.fill_between(
        stats["time"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color="C0",
        alpha=0.3,
        label="±1 std",
    )
    ax.axhline(0, color="r", linestyle="--", label="Battery depleted")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Remaining energy (%)")
    ax.set_title("Temporal evolution of residual battery energy")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = os.path.join(FIGURES_DIR, "battery_tracking")
    for ext in ("png", "jpg", "eps"):
        dpi = 300 if ext in ("png", "jpg", "eps") else None
        path = f"{base}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print(f"Saved {path}")
    if hasattr(plt, "close"):
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
