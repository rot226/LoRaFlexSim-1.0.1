from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step1_matrix  # noqa: E402  pylint: disable=wrong-import-position


def _mean_snir(row: dict[str, str]) -> float:
    histogram = json.loads(row["snr_histogram_json"])
    total = sum(histogram.values()) or 1
    return sum(float(bin_key) * count for bin_key, count in histogram.items()) / total


def _collect_metrics(results_dir: Path) -> dict[bool, list[dict[str, float]]]:
    grouped: dict[bool, list[dict[str, float]]] = {True: [], False: []}

    for csv_path in sorted(results_dir.glob("**/*.csv")):
        with csv_path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        grouped[use_snir].append(
            {
                "pdr": float(row["PDR"]),
                "collisions": float(row["collisions"]),
                "mean_snir": _mean_snir(row),
            }
        )

    return grouped


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_subset_metrics_stay_within_bounds(tmp_path: Path) -> None:
    results_dir = tmp_path / "subset"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "--nodes",
        "12",
        "24",
        "--packet-intervals",
        "5",
        "--duration",
        "20",
        "--results-dir",
        str(results_dir),
    ]

    run_step1_matrix.main(args)

    metrics = _collect_metrics(results_dir)
    assert metrics[True] and metrics[False], "Les états SNIR doivent être couverts"

    for use_snir, rows in metrics.items():
        pdr_mean = sum(item["pdr"] for item in rows) / len(rows)
        collisions_mean = sum(item["collisions"] for item in rows) / len(rows)
        snir_mean = sum(item["mean_snir"] for item in rows) / len(rows)

        assert 0.75 <= pdr_mean <= 1.01, f"PDR moyen inattendu pour SNIR={use_snir}: {pdr_mean:.3f}"
        assert 0.0 <= collisions_mean <= 10.0, (
            f"Taux de collisions moyen hors bornes pour SNIR={use_snir}: {collisions_mean:.3f}"
        )
        assert -10.0 <= snir_mean <= 40.0, (
            f"SNIR moyen irréaliste pour SNIR={use_snir}: {snir_mean:.2f} dB"
        )

