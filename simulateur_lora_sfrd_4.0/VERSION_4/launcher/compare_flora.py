"""Utilities to compare simulator results against FLoRa output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_flora_metrics(csv_path: str | Path) -> dict[str, Any]:
    """Return PDR and SF histogram from a FLoRa CSV export."""
    df = pd.read_csv(csv_path)
    total_sent = int(df["sent"].sum()) if "sent" in df.columns else 0
    total_recv = int(df["received"].sum()) if "received" in df.columns else 0
    pdr = total_recv / total_sent if total_sent else 0.0
    sf_cols = [c for c in df.columns if c.startswith("sf")]
    sf_hist = {int(c[2:]): int(df[c].sum()) for c in sf_cols}
    return {"PDR": pdr, "sf_distribution": sf_hist}


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
