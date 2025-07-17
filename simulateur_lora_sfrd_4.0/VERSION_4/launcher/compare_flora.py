"""Utilities to compare simulator results against FLoRa output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_flora_metrics(csv_path: str | Path) -> dict[str, Any]:
    """Return common metrics aggregated from a FLoRa CSV export."""

    df = pd.read_csv(csv_path)

    total_sent = int(df["sent"].sum()) if "sent" in df.columns else 0
    total_recv = int(df["received"].sum()) if "received" in df.columns else 0
    total_col = int(df["collisions"].sum()) if "collisions" in df.columns else 0
    pdr = total_recv / total_sent if total_sent else 0.0

    sf_cols = [c for c in df.columns if c.startswith("sf") and c[2:].isdigit()]
    sf_hist = {int(c[2:]): int(df[c].sum()) for c in sf_cols}

    energy_col = None
    for c in ("energy_J", "energy"):
        if c in df.columns:
            energy_col = c
            break
    total_energy = float(df[energy_col].sum()) if energy_col else 0.0

    throughput = float(df["throughput_bps"].mean()) if "throughput_bps" in df.columns else 0.0

    coll_cols = [c for c in df.columns if c.startswith("collisions_sf")]
    coll_dist = {int(c.split("sf")[1]): int(df[c].sum()) for c in coll_cols}

    return {
        "PDR": pdr,
        "sf_distribution": sf_hist,
        "throughput_bps": throughput,
        "energy_J": total_energy,
        "collisions": total_col,
        "collisions_distribution": coll_dist,
    }


def compare_with_sim(
    sim_metrics: dict[str, Any],
    flora_csv: str | Path,
    *,
    pdr_tol: float = 0.05,
    throughput_tol: float = 0.05,
    energy_tol: float = 0.05,
) -> bool:
    """Compare simulator metrics with FLoRa results.

    Parameters
    ----------
    sim_metrics : dict
        Metrics returned by :meth:`Simulator.get_metrics`.
    flora_csv : str | Path
        Path to the FLoRa CSV export to compare against.
    pdr_tol : float, optional
        Accepted absolute tolerance on the PDR difference. Defaults to ``0.05``.

    Returns
    -------
    bool
        ``True`` if the metrics match within tolerance.
    """
    flora_metrics = load_flora_metrics(flora_csv)

    pdr_match = abs(sim_metrics.get("PDR", 0.0) - flora_metrics["PDR"]) <= pdr_tol

    sf_match = sim_metrics.get("sf_distribution") == flora_metrics["sf_distribution"]

    th_match = True
    if "throughput_bps" in sim_metrics and flora_metrics.get("throughput_bps"):
        ref = max(flora_metrics["throughput_bps"], 1.0)
        th_match = (
            abs(sim_metrics["throughput_bps"] - flora_metrics["throughput_bps"])
            <= throughput_tol * ref
        )

    energy_match = True
    if "energy_J" in sim_metrics and flora_metrics.get("energy_J"):
        ref = max(flora_metrics["energy_J"], 1.0)
        energy_match = (
            abs(sim_metrics["energy_J"] - flora_metrics["energy_J"])
            <= energy_tol * ref
        )

    coll_match = True
    if "collisions_distribution" in sim_metrics and flora_metrics.get("collisions_distribution"):
        coll_match = (
            sim_metrics["collisions_distribution"]
            == flora_metrics["collisions_distribution"]
        )

    return pdr_match and sf_match and th_match and energy_match and coll_match
