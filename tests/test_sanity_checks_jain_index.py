import csv
import json
from pathlib import Path

from scripts.sanity_checks import run_checks


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(*, snir_state: str, jain_index: str) -> dict[str, str]:
    distribution = json.dumps({"7": 1, "8": 2})
    collision_breakdown = json.dumps({"by_sf": {"7": 1, "8": 2}})
    return {
        "algorithm": "algo",
        "num_nodes": "10",
        "packet_interval_s": "1.0",
        "simulation_duration_s": "100.0",
        "snir_state": snir_state,
        "PDR": "0.6" if snir_state == "snir_on" else "0.4",
        "throughput_bps": "1000" if snir_state == "snir_on" else "800",
        "collisions": "2" if snir_state == "snir_on" else "5",
        "sf_distribution_json": distribution,
        "snr_histogram_json": distribution,
        "snir_histogram_json": distribution,
        "collision_breakdown_json": collision_breakdown,
        "jain_index": jain_index,
    }


def test_jain_index_warn_absent_when_not_all_ones(tmp_path: Path) -> None:
    rows = [
        _base_row(snir_state="snir_on", jain_index="1.0"),
        _base_row(snir_state="snir_off", jain_index="0.9"),
    ]
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, rows)

    warnings, failures = run_checks(
        [csv_path],
        epsilon=0.01,
        large_nodes=100,
        high_pdr_threshold=0.999,
    )

    assert not failures
    assert not any(
        issue.message == "Indice de Jain == 1.0 pour toutes les lignes."
        for issue in warnings
    )


def test_jain_index_warn_present_when_all_ones(tmp_path: Path) -> None:
    rows = [
        _base_row(snir_state="snir_on", jain_index="1.0"),
        _base_row(snir_state="snir_off", jain_index="1.0"),
    ]
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, rows)

    warnings, failures = run_checks(
        [csv_path],
        epsilon=0.01,
        large_nodes=100,
        high_pdr_threshold=0.999,
    )

    assert not failures
    assert any(
        issue.message == "Indice de Jain == 1.0 pour toutes les lignes."
        for issue in warnings
    )
