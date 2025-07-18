"""Utilities to compare simulator results against FLoRa output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_flora_metrics(csv_path: str | Path) -> dict[str, Any]:
    """Return metrics from a FLoRa CSV export.

    The file is expected to contain at least the columns ``sent`` and ``received``
    to compute the PDR as well as ``sfX`` columns describing the spreading factor
    distribution.  Additional optional columns may be present:

    ``throughput_bps``
        Average throughput in bits per second.
    ``energy`` or ``energy_J``
        Energy consumption for the run.
    ``collisions``
        Total number of packet collisions.
    ``collisions_sfX``
        Number of collisions that occurred with spreading factor ``X``.
    """
    df = pd.read_csv(csv_path)
    total_sent = int(df["sent"].sum()) if "sent" in df.columns else 0
    total_recv = int(df["received"].sum()) if "received" in df.columns else 0
    pdr = total_recv / total_sent if total_sent else 0.0
    sf_cols = [c for c in df.columns if c.startswith("sf") and not c.startswith("sf_collisions")]
    sf_hist = {int(c[2:]): int(df[c].sum()) for c in sf_cols}

    throughput = float(df["throughput_bps"].mean()) if "throughput_bps" in df.columns else 0.0
    if "energy_J" in df.columns:
        energy = float(df["energy_J"].mean())
    elif "energy" in df.columns:
        energy = float(df["energy"].mean())
    else:
        energy = 0.0

    collision_cols = [c for c in df.columns if c.startswith("collisions_sf")]
    collision_dist = {int(c.split("sf")[1]): int(df[c].sum()) for c in collision_cols}

    return {
        "PDR": pdr,
        "sf_distribution": sf_hist,
        "throughput_bps": throughput,
        "energy_J": energy,
        "collision_distribution": collision_dist,
    }


def compare_with_sim(sim_metrics: dict[str, Any], flora_csv: str | Path, *, pdr_tol: float = 0.05) -> bool:
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
    return pdr_match and sf_match


def load_flora_rx_stats(csv_path: str | Path) -> dict[str, Any]:
    """Load average RSSI/SNR and collisions from a FLoRa export."""
    df = pd.read_csv(csv_path)
    rssi = float(df["rssi"].mean()) if "rssi" in df.columns else 0.0
    snr = float(df["snr"].mean()) if "snr" in df.columns else 0.0
    collisions = int(df["collisions"].sum()) if "collisions" in df.columns else 0
    return {"rssi": rssi, "snr": snr, "collisions": collisions}
