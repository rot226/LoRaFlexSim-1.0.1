from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _read_cdf(path: Path) -> dict[str, list[tuple[float, float]]]:
    by_window: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            window = row["window"]
            by_window.setdefault(window, []).append(
                (float(row["snir_db"]), float(row["cdf"]))
            )
    return {key: sorted(values) for key, values in by_window.items()}


def _quantile_from_cdf(points: list[tuple[float, float]], q: float) -> float:
    for snir_db, cdf in points:
        if cdf >= q:
            return snir_db
    return points[-1][0]


def test_snir_window_sensitivity(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "experiments" / "snir_stage1" / "scenarios" / "snir_window_cdf.py"
    output_csv = tmp_path / "snir_window_cdf.csv"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--window-modes",
            "packet,preamble",
            "--num-nodes",
            "500",
            "--packets-per-node",
            "4",
            "--packet-interval",
            "300",
            "--output",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    cdf_by_window = _read_cdf(output_csv)
    assert {"packet", "preamble"}.issubset(cdf_by_window.keys())

    packet_cdf = cdf_by_window["packet"]
    preamble_cdf = cdf_by_window["preamble"]

    packet_median = _quantile_from_cdf(packet_cdf, 0.5)
    preamble_median = _quantile_from_cdf(preamble_cdf, 0.5)
    packet_iqr = _quantile_from_cdf(packet_cdf, 0.75) - _quantile_from_cdf(packet_cdf, 0.25)
    preamble_iqr = _quantile_from_cdf(preamble_cdf, 0.75) - _quantile_from_cdf(preamble_cdf, 0.25)

    median_gap = abs(packet_median - preamble_median)
    dispersion_gap = abs(packet_iqr - preamble_iqr)

    assert median_gap >= 2.0 or dispersion_gap >= 3.0, (
        f"Les fenêtres SNIR doivent diverger: Δ médiane={median_gap:.2f} dB, "
        f"Δ dispersion={dispersion_gap:.2f} dB."
    )
