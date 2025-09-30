"""Utilities to handle non-orthogonal capture threshold matrices.

This module exposes the default matrix used to model the capture effect
between different spreading factors.  It also provides a helper to load a
matrix from JSON or INI files so that simulations can easily try alternative
values derived from measurements.
"""

from __future__ import annotations

import json
import configparser
from pathlib import Path
from typing import List

# Default FLoRa matrix taken from the original C++ implementation
DEFAULT_NON_ORTH_DELTA: List[List[float]] = [
    [1, -8, -9, -9, -9, -9],
    [-11, 1, -11, -12, -13, -13],
    [-15, -13, 1, -13, -14, -15],
    [-19, -18, -17, 1, -17, -18],
    [-22, -22, -21, -20, 1, -20],
    [-25, -25, -25, -24, -23, 1],
]


def load_non_orth_delta(path: str | None) -> List[List[float]]:
    """Load a 6x6 capture threshold matrix from *path*.

    ``path`` may point to a JSON file containing a list of six lists or to an
    INI file with a ``[NON_ORTH_DELTA]`` section.  INI rows should be named
    ``SF7`` to ``SF12`` and contain six comma or space separated numbers.

    If *path* is ``None`` or the file cannot be read, the
    :data:`DEFAULT_NON_ORTH_DELTA` matrix is returned.
    """

    def _clone_default() -> List[List[float]]:
        """Return a defensive copy of :data:`DEFAULT_NON_ORTH_DELTA`."""

        return [row.copy() for row in DEFAULT_NON_ORTH_DELTA]

    if path is None:
        return _clone_default()

    p = Path(path)
    try:
        if p.suffix.lower() == ".json":
            with p.open("r", encoding="utf8") as f:
                data = json.load(f)
            # basic validation
            if not (isinstance(data, list) and len(data) == 6):
                raise ValueError("JSON matrix must be a list of 6 lists")
            return [list(map(float, row)) for row in data]

        if p.suffix.lower() in {".ini", ".cfg"}:
            cp = configparser.ConfigParser()
            cp.read(p)
            matrix: List[List[float]] = []
            for sf in range(7, 13):
                key = f"SF{sf}"
                raw = cp.get("NON_ORTH_DELTA", key, fallback=None)
                if raw is None:
                    matrix.append(DEFAULT_NON_ORTH_DELTA[sf - 7].copy())
                    continue
                parts = [x for x in raw.replace(",", " ").split() if x]
                if len(parts) != 6:
                    raise ValueError("Each INI row must contain 6 values")
                matrix.append([float(x) for x in parts])
            return matrix
    except Exception:
        # Fall back to default on any error to keep behaviour robust
        return _clone_default()

    raise ValueError("Unsupported file format; use JSON or INI")
