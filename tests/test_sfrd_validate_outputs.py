from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

EXPECTED_SIZES = [80, 160, 320, 640, 1280]
EXPECTED_ALGOS = ["UCB", "ADR", "MixRA-H", "MixRA-Opt"]
EXPECTED_SNIR = ["OFF", "ON"]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _seed_valid_output(root: Path) -> None:
    for snir in EXPECTED_SNIR:
        rows_pdr: list[list[object]] = []
        rows_thr: list[list[object]] = []
        rows_energy: list[list[object]] = []
        rows_sf: list[list[object]] = []
        for size in EXPECTED_SIZES:
            for algo in EXPECTED_ALGOS:
                rows_pdr.append([size, algo, snir, 0.75])
                rows_thr.append([size, algo, snir, 1.2])
                rows_energy.append([size, algo, snir, 0.5])
                rows_sf.append([size, algo, snir, 7, 10])

        _write_csv(root / f"SNIR_{snir}" / "pdr_results.csv", ["network_size", "algorithm", "snir", "pdr"], rows_pdr)
        _write_csv(
            root / f"SNIR_{snir}" / "throughput_results.csv",
            ["network_size", "algorithm", "snir", "throughput_packets_per_s"],
            rows_thr,
        )
        _write_csv(
            root / f"SNIR_{snir}" / "energy_results.csv",
            ["network_size", "algorithm", "snir", "energy_joule_per_packet"],
            rows_energy,
        )
        _write_csv(
            root / f"SNIR_{snir}" / "sf_distribution.csv",
            ["network_size", "algorithm", "snir", "sf", "count"],
            rows_sf,
        )

    _write_csv(root / "learning_curve_ucb.csv", ["episode", "reward"], [[1, 0.2], [2, 0.3]])


def test_validate_outputs_generates_release_report(tmp_path: Path) -> None:
    output_root = tmp_path / "sfrd_output"
    _seed_valid_output(output_root)

    result = subprocess.run(
        [sys.executable, "-m", "sfrd.cli.validate_outputs", "--output-root", str(output_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report_path = output_root / "release_report.txt"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Runs attendus: 40" in report
    assert "Runs réalisés: 40" in report
    assert "Anomalies détectées:" in report
    assert "- Aucune" in report


def test_validate_outputs_blocks_release_on_matrix_incomplete(tmp_path: Path) -> None:
    output_root = tmp_path / "sfrd_output"
    _seed_valid_output(output_root)

    pdr_path = output_root / "SNIR_OFF" / "pdr_results.csv"
    rows = pdr_path.read_text(encoding="utf-8").splitlines()
    pdr_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "sfrd.cli.validate_outputs", "--output-root", str(output_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    report = (output_root / "release_report.txt").read_text(encoding="utf-8")
    assert "matrix_completeness" in report


def test_validate_outputs_partial_mode_is_diagnostic(tmp_path: Path) -> None:
    output_root = tmp_path / "sfrd_output"
    _seed_valid_output(output_root)

    pdr_path = output_root / "SNIR_OFF" / "pdr_results.csv"
    rows = pdr_path.read_text(encoding="utf-8").splitlines()
    pdr_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfrd.cli.validate_outputs",
            "--output-root",
            str(output_root),
            "--mode",
            "partial",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    report = (output_root / "release_report.txt").read_text(encoding="utf-8")
    assert "[warning] (matrix_completeness)" in report
