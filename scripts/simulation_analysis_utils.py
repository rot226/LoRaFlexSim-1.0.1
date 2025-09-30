"""Utility helpers for simulation analysis scripts.

This module centralises filesystem helpers and statistical utilities
used by multiple scenario automation scripts.  It keeps the scripts
lightweight while ensuring consistent computation of latency and energy
statistics.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def ensure_directory(path: str) -> None:
    """Create *path* if it does not already exist."""

    os.makedirs(path, exist_ok=True)


def ensure_output_directories() -> None:
    """Create the standard ``results`` and ``figures`` folders."""

    ensure_directory(RESULTS_DIR)
    ensure_directory(FIGURES_DIR)


def collect_latencies(
    events: Sequence[dict],
    node_ids: Iterable[int] | None = None,
) -> list[float]:
    """Return transmission delays for successful packets.

    Parameters
    ----------
    events:
        Sequence of event dictionaries produced by ``Simulator``.
    node_ids:
        Optional iterable restricting the collection to specific nodes.
    """

    node_filter = set(node_ids) if node_ids is not None else None
    latencies: list[float] = []
    for entry in events:
        if entry.get("result") != "Success":
            continue
        if node_filter is not None and entry.get("node_id") not in node_filter:
            continue
        start = float(entry.get("start_time", 0.0))
        end = float(entry.get("end_time", start))
        latencies.append(end - start)
    return latencies


def summarise_nodes(
    nodes: Sequence[object],
    node_ids: Iterable[int] | None = None,
) -> tuple[int, int, list[float]]:
    """Return attempts, deliveries and per-node energy for the subset."""

    node_filter = set(node_ids) if node_ids is not None else None
    attempts = 0
    deliveries = 0
    energies: list[float] = []
    for node in nodes:
        if node_filter is not None and getattr(node, "id", None) not in node_filter:
            continue
        attempts += int(getattr(node, "tx_attempted", 0))
        deliveries += int(getattr(node, "rx_delivered", 0))
        energies.append(float(getattr(node, "energy_consumed", 0.0)))
    return attempts, deliveries, energies


def compute_latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    """Compute mean and 95th percentile latency."""

    if not latencies:
        return {"mean": 0.0, "p95": 0.0}
    try:
        arr = np.asarray(latencies, dtype=float)
    except TypeError:  # numpy_stub compatibility
        arr = np.asarray(latencies)
    if hasattr(arr, "mean"):
        mean_val = float(arr.mean())
    else:
        seq = list(arr)
        mean_val = float(sum(seq) / len(seq))
    try:
        p95_val = float(np.percentile(arr, 95))
    except AttributeError:
        seq = sorted(float(x) for x in arr)
        if not seq:
            p95_val = 0.0
        else:
            idx = max(0, min(len(seq) - 1, int(math.ceil(0.95 * len(seq)) - 1)))
            p95_val = seq[idx]
    return {"mean": mean_val, "p95": p95_val}


def place_gateways_in_grid(gateways: Sequence[object], area_size: float) -> None:
    """Reposition gateways on a uniform grid covering the area."""

    count = len(gateways)
    if count == 0:
        return
    side = int(math.ceil(math.sqrt(count)))
    spacing = area_size / (side + 1)
    for index, gateway in enumerate(gateways):
        row = index // side
        col = index % side
        setattr(gateway, "x", spacing * (col + 1))
        setattr(gateway, "y", spacing * (row + 1))


def _select_columns(records: Iterable[dict], columns: Sequence[str]) -> Iterator[dict]:
    for record in records:
        yield {col: record.get(col) for col in columns}


def export_lightweight_trace(
    records: Sequence[dict],
    destination: str | os.PathLike,
    columns: Sequence[str],
    *,
    fmt: str | None = None,
    compression: str | None = None,
) -> Path:
    """Export ``records`` keeping only ``columns`` in a compact format."""

    path = Path(destination)
    suffixes = [s.lower() for s in path.suffixes]
    if fmt is None:
        if suffixes and suffixes[-1] == ".parquet":
            fmt = "parquet"
        else:
            fmt = "csv"
    fmt = fmt.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(_select_columns(records, columns))

    if fmt == "csv":
        use_gzip = False
        if compression is None:
            use_gzip = suffixes[-1:] in ([".gz"], [".gzip"])
        elif compression.lower() in {"gz", "gzip"}:
            use_gzip = True
        opener = gzip.open if use_gzip else open
        mode = "wt"
        with opener(path, mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in selected:
                writer.writerow(row)
        return path
    if fmt == "parquet":
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required for Parquet export") from exc
        frame = pd.DataFrame(selected, columns=list(columns))
        frame.to_parquet(path, index=False, compression=compression)
        return path
    raise ValueError(f"Unsupported trace format: {fmt}")


def cache_metrics_ready(
    events: Sequence[dict],
    nodes: Sequence[object],
    destination: str | os.PathLike,
) -> Path:
    """Persist pre-aggregated metrics for quick plotting scripts."""

    latencies = collect_latencies(events)
    stats = compute_latency_stats(latencies)
    attempts, deliveries, energies = summarise_nodes(nodes)
    pdr = (deliveries / attempts) if attempts else 0.0
    if energies:
        try:
            mean_energy = float(np.mean(energies))
        except AttributeError:
            mean_energy = float(sum(energies) / len(energies))
    else:
        mean_energy = 0.0
    payload = {
        "attempts": attempts,
        "deliveries": deliveries,
        "pdr": pdr,
        "latency_mean": stats["mean"],
        "latency_p95": stats["p95"],
        "mean_energy": mean_energy,
        "node_count": len(nodes),
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
