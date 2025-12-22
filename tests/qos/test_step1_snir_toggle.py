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

import run_step1_matrix as step1_matrix


MIN_SNIR_DELTA_DB = 4.0
MIN_DER_PDR_DELTA = 0.06
MIN_COLLISION_DELTA = 5


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_toggle_generates_distinct_csv(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "2",
        "--nodes",
        "300",
        "--packet-intervals",
        "0.1",
        "--duration",
        "60",
        "--results-dir",
        str(results_dir),
    ]

    step1_matrix.main(args)

    csv_paths = sorted(results_dir.glob("**/*.csv"))
    assert csv_paths, "Aucun CSV généré par run_step1_matrix"

    snir_states: set[str] = set()
    mean_snir_by_state: dict[bool, list[float]] = {True: [], False: []}
    der_by_state: dict[bool, list[float]] = {True: [], False: []}
    pdr_by_state: dict[bool, list[float]] = {True: [], False: []}
    collisions_by_state: dict[bool, list[float]] = {True: [], False: []}

    for path in csv_paths:
        assert path.name.endswith(("_snir-on.csv", "_snir-off.csv")), path.name

        with path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        snir_state = row["snir_state"]
        snir_states.add(snir_state)

        histogram = json.loads(row["snr_histogram_json"])
        total_samples = sum(histogram.values()) or 1
        mean_snir = sum(float(bin_key) * count for bin_key, count in histogram.items()) / total_samples
        mean_snir_by_state[use_snir].append(mean_snir)

        der_by_state[use_snir].append(float(row["DER"]))
        pdr_by_state[use_snir].append(float(row["PDR"]))
        collisions_by_state[use_snir].append(float(row["collisions"]))

    assert snir_states == {"snir_on", "snir_off"}

    mean_on = sum(mean_snir_by_state[True]) / len(mean_snir_by_state[True])
    mean_off = sum(mean_snir_by_state[False]) / len(mean_snir_by_state[False])
    assert abs(mean_on - mean_off) >= MIN_SNIR_DELTA_DB

    avg_der_on = sum(der_by_state[True]) / len(der_by_state[True])
    avg_der_off = sum(der_by_state[False]) / len(der_by_state[False])
    avg_pdr_on = sum(pdr_by_state[True]) / len(pdr_by_state[True])
    avg_pdr_off = sum(pdr_by_state[False]) / len(pdr_by_state[False])
    avg_collisions_on = sum(collisions_by_state[True]) / len(collisions_by_state[True])
    avg_collisions_off = sum(collisions_by_state[False]) / len(collisions_by_state[False])

    assert abs(avg_der_on - avg_der_off) >= MIN_DER_PDR_DELTA
    assert abs(avg_pdr_on - avg_pdr_off) >= MIN_DER_PDR_DELTA
    assert abs(avg_collisions_on - avg_collisions_off) >= MIN_COLLISION_DELTA
