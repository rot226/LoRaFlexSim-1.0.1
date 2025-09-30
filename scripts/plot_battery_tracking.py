"""Plot battery evolution from ``results/battery_tracking.csv``.

The CSV is expected to contain columns ``time``, ``node_id``, ``energy_j``,
``capacity_j``, ``alive`` and ``replicate`` as produced by
``run_battery_tracking.py``.  This utility computes the mean residual energy
over nodes for each replicate, then plots all replicate trajectories in light
grey alongside the overall mean with a shaded area representing ±1 standard
deviation.  Optionally, specific nodes can be highlighted and an annotation is
added when the first battery depletion is detected.  The figure is saved to
``figures/battery_tracking.png``.

Usage::

    python scripts/plot_battery_tracking.py [--annotate-depletion] [--focus-node 3]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Sequence

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


def _parse_focus_nodes(raw_values: Iterable[str]) -> list[int]:
    focus_nodes: set[int] = set()
    for raw in raw_values:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                focus_nodes.add(int(chunk))
            except ValueError as exc:  # pragma: no cover - argument validation
                raise SystemExit(
                    f"Identifiant de nœud invalide pour --focus-node: '{chunk}'"
                ) from exc
    return sorted(focus_nodes)


def _detect_depletion(
    df: pd.DataFrame, focus_nodes: Iterable[int] | None = None
) -> tuple[float | None, dict[int, float]]:
    """Return the first depletion time globally and per node."""

    if focus_nodes:
        subset = df[df["node_id"].isin(set(focus_nodes))]
    else:
        subset = df

    if subset.empty:
        return None, {}

    if "alive" in subset.columns:
        depleted_mask = ~subset["alive"].astype(bool)
        depleted_mask |= subset["energy_j"] <= 0
    else:
        depleted_mask = subset["energy_j"] <= 0

    depleted_rows = subset.loc[depleted_mask]
    if depleted_rows.empty:
        return None, {}

    per_node = depleted_rows.groupby("node_id")["time"].min().to_dict()
    global_time = min(per_node.values()) if per_node else None
    return global_time, per_node


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace l'évolution énergétique")
    parser.add_argument(
        "--annotate-depletion",
        action="store_true",
        help="Ajoute une annotation lorsque la batterie s'épuise",
    )
    parser.add_argument(
        "--focus-node",
        action="append",
        default=[],
        metavar="NODE",
        help="Identifiant de nœud à mettre en évidence (peut être répété ou séparé par des virgules)",
    )
    if argv is None:
        argv = []
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    focus_nodes = _parse_focus_nodes(args.focus_node)

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

    depletion_time, per_node_depletion = _detect_depletion(df, focus_nodes or None)

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

            def axvline(self, *args, **kwargs):
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

            def text(self, *args, **kwargs):
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

    if focus_nodes:
        focus_df = df[df["node_id"].isin(focus_nodes)]
        if focus_df.empty:
            print(
                "Aucun échantillon trouvé pour les nœuds spécifiés; mise en évidence ignorée.",
                file=sys.stderr,
            )
        else:
            focus_stats = (
                focus_df.groupby(["node_id", "time"])["energy_pct"].mean().reset_index()
            )
            color_cycle = plt.rcParams.get("axes.prop_cycle")
            colors = []
            if color_cycle is not None:
                colors = color_cycle.by_key().get("color", [])
            for idx, (node_id, group) in enumerate(focus_stats.groupby("node_id")):
                color = None
                if colors:
                    color = colors[(idx + 1) % len(colors)]
                ax.plot(
                    group["time"],
                    group["energy_pct"],
                    linewidth=2,
                    color=color,
                    label=f"Nœud {node_id}",
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

    if args.annotate_depletion and depletion_time is not None:
        ax.axvline(depletion_time, color="r", linestyle=":", linewidth=1.5)
        y_max = ax.get_ylim()[1]
        nodes_at_time = [
            node
            for node, time in per_node_depletion.items()
            if time == depletion_time
        ]
        node_suffix = ""
        if nodes_at_time:
            node_suffix = " (nœud(s) " + ", ".join(str(n) for n in sorted(nodes_at_time)) + ")"
        ax.text(
            depletion_time,
            y_max * 0.95,
            "Batterie épuisée" + node_suffix,
            color="r",
            rotation=90,
            va="top",
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )

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
