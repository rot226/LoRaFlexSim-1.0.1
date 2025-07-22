"""Utilities to compare simulator results against FLoRa output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # pandas is optional when simply parsing raw simulator metrics
    import pandas as pd
except Exception:  # pragma: no cover - pandas may not be installed
    pd = None


def _parse_sca_file(path: Path) -> dict[str, Any]:
    """Return raw metrics from an OMNeT++ ``.sca`` file.

    The resulting dictionary can be converted to a DataFrame to compute
    aggregates similar to a FLoRa CSV export.
    """
    metrics: dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "scalar":
                name = parts[2]
                try:
                    value = float(parts[3])
                except ValueError:
                    continue
                metrics[name] = metrics.get(name, 0.0) + value

    row: dict[str, Any] = {}
    for key, val in metrics.items():
        if key in {"sent", "received", "collisions"}:
            row[key] = int(val)
        elif key in {"throughput_bps", "energy_J", "energy", "rssi", "snr", "avg_delay_s"}:
            row[key] = float(val)
        elif key.startswith("sf") and key[2:].isdigit():
            row[key] = int(val)
        elif key.startswith("collisions_sf"):
            row[key] = int(val)
        elif key.startswith("energy_class_"):
            row[key] = float(val)

    return row


def _aggregate_df(df: 'pd.DataFrame') -> dict[str, Any]:
    """Compute aggregated metrics from a DataFrame of raw values."""
    if pd is None:
        raise RuntimeError("pandas is required for this function")
    total_sent = int(df["sent"].sum()) if "sent" in df.columns else 0
    total_recv = int(df["received"].sum()) if "received" in df.columns else 0
    pdr = total_recv / total_sent if total_sent else 0.0

    sf_cols = [
        c
        for c in df.columns
        if c.startswith("sf") and not c.startswith("collisions_sf")
    ]
    sf_hist = {int(c[2:]): int(df[c].sum()) for c in sf_cols}

    throughput = (
        float(df["throughput_bps"].mean()) if "throughput_bps" in df.columns else 0.0
    )
    avg_delay = float(df["avg_delay_s"].mean()) if "avg_delay_s" in df.columns else 0.0
    if "energy_J" in df.columns:
        energy = float(df["energy_J"].mean())
    elif "energy" in df.columns:
        energy = float(df["energy"].mean())
    else:
        energy = 0.0

    collisions = int(df["collisions"].sum()) if "collisions" in df.columns else 0
    collision_cols = [c for c in df.columns if c.startswith("collisions_sf")]
    collision_dist = {int(c.split("sf")[1]): int(df[c].sum()) for c in collision_cols}

    energy_class = {
        col.split("_")[2]: float(df[col].mean())
        for col in df.columns
        if col.startswith("energy_class_")
    }

    return {
        "PDR": pdr,
        "sf_distribution": sf_hist,
        "throughput_bps": throughput,
        "energy_J": energy,
        "avg_delay_s": avg_delay,
        **{f"energy_class_{cls}_J": val for cls, val in energy_class.items()},
        "collision_distribution": collision_dist,
        "collisions": collisions,
    }


def _load_sca_file(path: Path) -> dict[str, Any]:
    """Parse a single ``.sca`` file and compute aggregated metrics."""
    if pd is None:
        raise RuntimeError("pandas is required for this function")
    row = _parse_sca_file(path)
    df = pd.DataFrame([row])
    return _aggregate_df(df)


def load_flora_metrics(path: str | Path) -> dict[str, Any]:
    """Return metrics from a FLoRa export.

    The path can point to a CSV converted from the original ``.sca``/``.vec``
    files produced by OMNeT++. The CSV is expected to contain at least the
    columns ``sent`` and ``received`` to compute the PDR as well as ``sfX``
    columns describing the spreading factor distribution. Additional optional
    columns may be present:

    ``throughput_bps``
        Average throughput in bits per second.
    ``energy`` or ``energy_J``
        Energy consumption for the run.
    ``collisions``
        Total number of packet collisions.
    ``collisions_sfX``
        Number of collisions that occurred with spreading factor ``X``.
    """
    if pd is None:
        raise RuntimeError("pandas is required for this function")
    path = Path(path)
    if path.is_dir():
        rows = [_parse_sca_file(p) for p in sorted(path.glob("*.sca"))]
        df = pd.DataFrame(rows)
        return _aggregate_df(df)

    if path.suffix.lower() == ".sca":
        return _load_sca_file(path)

    df = pd.read_csv(path)
    total_sent = int(df["sent"].sum()) if "sent" in df.columns else 0
    total_recv = int(df["received"].sum()) if "received" in df.columns else 0
    pdr = total_recv / total_sent if total_sent else 0.0
    sf_cols = [
        c
        for c in df.columns
        if c.startswith("sf") and not c.startswith("sf_collisions")
    ]
    sf_hist = {int(c[2:]): int(df[c].sum()) for c in sf_cols}

    throughput = (
        float(df["throughput_bps"].mean()) if "throughput_bps" in df.columns else 0.0
    )
    avg_delay = float(df["avg_delay_s"].mean()) if "avg_delay_s" in df.columns else 0.0
    if "energy_J" in df.columns:
        energy = float(df["energy_J"].mean())
    elif "energy" in df.columns:
        energy = float(df["energy"].mean())
    else:
        energy = 0.0

    collisions = int(df["collisions"].sum()) if "collisions" in df.columns else 0

    collision_cols = [c for c in df.columns if c.startswith("collisions_sf")]
    collision_dist = {int(c.split("sf")[1]): int(df[c].sum()) for c in collision_cols}
    energy_class = {
        col.split("_")[2]: float(df[col].mean())
        for col in df.columns
        if col.startswith("energy_class_")
    }

    return {
        "PDR": pdr,
        "sf_distribution": sf_hist,
        "throughput_bps": throughput,
        "energy_J": energy,
        "avg_delay_s": avg_delay,
        **{f"energy_class_{cls}_J": val for cls, val in energy_class.items()},
        "collision_distribution": collision_dist,
        "collisions": collisions,
    }


def compare_with_sim(
    sim_metrics: dict[str, Any], flora_csv: str | Path, *, pdr_tol: float = 0.05
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
    return pdr_match and sf_match


def load_flora_rx_stats(path: str | Path) -> dict[str, Any]:
    """Load average RSSI/SNR and collisions from a FLoRa export."""
    if pd is None:
        raise RuntimeError("pandas is required for this function")
    path = Path(path)
    if path.is_dir():
        rows = [_parse_sca_file(p) for p in sorted(path.glob("*.sca"))]
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".sca":
        df = pd.DataFrame([_parse_sca_file(path)])
    else:
        df = pd.read_csv(path)

    rssi = float(df["rssi"].mean()) if "rssi" in df.columns else 0.0
    snr = float(df["snr"].mean()) if "snr" in df.columns else 0.0
    collisions = int(df["collisions"].sum()) if "collisions" in df.columns else 0
    return {"rssi": rssi, "snr": snr, "collisions": collisions}
