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


MIN_MEAN_SNIR_DELTA_DB = 4.5
MIN_FADING_GAP_GAIN_DB = 0.5
FADING_STD_DB = 3.0
NO_FADING_STD_DB = 0.0


def _mean_snir_from_row(row: dict[str, str]) -> float:
    histogram = json.loads(row["snir_histogram_json"])
    total = sum(histogram.values()) or 1
    return sum(float(bin_key) * count for bin_key, count in histogram.items()) / total


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_delta_across_seeds(tmp_path: Path) -> None:
    def _run_gap(*, fading_std_db: float, label: str) -> float:
        results_dir = tmp_path / f"step1_snir_delta_{label}"
        run_step1_matrix.main(
            [
                "--algos",
                "adr",
                "--with-snir",
                "true",
                "false",
                "--seeds",
                "1",
                "2",
                "3",
                "--nodes",
                "240",
                "--packet-intervals",
                "0.2",
                "--duration",
                "45",
                "--fading-std-db",
                str(fading_std_db),
                "--results-dir",
                str(results_dir),
            ]
        )

        mean_snir_by_state: dict[bool, list[float]] = {True: [], False: []}
        for csv_path in sorted(results_dir.glob("**/*.csv")):
            with csv_path.open(newline="", encoding="utf8") as handle:
                row = next(csv.DictReader(handle))
            use_snir = row["use_snir"] == "True"
            mean_snir_by_state[use_snir].append(_mean_snir_from_row(row))

        assert (
            mean_snir_by_state[True] and mean_snir_by_state[False]
        ), "Les deux états SNIR sont requis"

        mean_on = sum(mean_snir_by_state[True]) / len(mean_snir_by_state[True])
        mean_off = sum(mean_snir_by_state[False]) / len(mean_snir_by_state[False])
        return abs(mean_on - mean_off)

    gap_no_fading = _run_gap(fading_std_db=NO_FADING_STD_DB, label="no_fading")
    gap_with_fading = _run_gap(fading_std_db=FADING_STD_DB, label="with_fading")

    assert gap_with_fading >= MIN_MEAN_SNIR_DELTA_DB, (
        f"Écart moyen de SNIR insuffisant entre états avec fading: {gap_with_fading:.2f} dB"
    )
    assert (gap_with_fading - gap_no_fading) >= MIN_FADING_GAP_GAIN_DB, (
        "Le fading doit accentuer l'écart SNIR on/off "
        f"(Δgain={gap_with_fading - gap_no_fading:.2f} dB)"
    )
