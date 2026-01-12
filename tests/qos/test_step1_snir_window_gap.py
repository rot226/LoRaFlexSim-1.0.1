from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step1_matrix  # noqa: E402  pylint: disable=wrong-import-position


MIN_MEAN_SNIR_WINDOW_GAP_DB = 1.5
FADING_STD_DB = 3.0


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_window_gap(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1_window_gap"
    run_step1_matrix.main(
        [
            "--algos",
            "adr",
            "--with-snir",
            "true",
            "--seeds",
            "1",
            "--nodes",
            "300",
            "--packet-intervals",
            "0.3",
            "--duration",
            "45",
            "--snir-windows",
            "packet",
            "preamble",
            "--fading-std-db",
            str(FADING_STD_DB),
            "--results-dir",
            str(results_dir),
        ]
    )

    snir_by_window: dict[str, list[float]] = {}
    for csv_path in sorted(results_dir.glob("**/*.csv")):
        with csv_path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))
        window = row.get("snir_window")
        if not window:
            continue
        snir_by_window.setdefault(window, []).append(float(row["snir_mean"]))

    assert "packet" in snir_by_window and "preamble" in snir_by_window, (
        "Les fenêtres packet et preamble doivent être présentes dans les CSV."
    )

    mean_packet = sum(snir_by_window["packet"]) / len(snir_by_window["packet"])
    mean_preamble = sum(snir_by_window["preamble"]) / len(snir_by_window["preamble"])
    gap = abs(mean_packet - mean_preamble)

    assert gap >= MIN_MEAN_SNIR_WINDOW_GAP_DB, (
        "Écart moyen de SNIR insuffisant entre les fenêtres "
        f"packet et preamble: {gap:.2f} dB"
    )
