from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"CSV vide : {path}"
    return rows[0]


def test_compare_generates_differences(tmp_path: Path) -> None:
    outdir = tmp_path / "snir_compare"
    script = Path(__file__).resolve().parents[2] / "experiments" / "snir_stage1_compare" / "scenarios" / "run_compare_stage1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--algorithms",
            "interference_only,snir_interference",
            "--profiles",
            "flora_full",
            "--nodes",
            "20",
            "--intervals",
            "1.0",
            "--reps",
            "1",
            "--jobs",
            "1",
            "--seed",
            "1",
            "--outdir",
            str(outdir),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    baseline_csv = outdir / "interference_only_compare.csv"
    snir_csv = outdir / "snir_interference_compare.csv"
    assert baseline_csv.exists(), f"Fichier manquant : {baseline_csv}"
    assert snir_csv.exists(), f"Fichier manquant : {snir_csv}"

    baseline = _read_single_row(baseline_csv)
    snir = _read_single_row(snir_csv)

    baseline_der = float(baseline["der"])
    snir_der = float(snir["der"])
    baseline_collisions = int(baseline["collisions"])
    snir_collisions = int(snir["collisions"])
    baseline_snir = float(baseline["snir_mean"])
    snir_snir = float(snir["snir_mean"])

    assert baseline_der != snir_der, "DER identiques entre modes SNIR on/off"
    assert baseline_collisions != snir_collisions, "Collisions identiques entre modes SNIR on/off"
    assert baseline_snir != snir_snir, "SNIR moyen identique entre modes SNIR on/off"
    assert any(outdir.iterdir()), "Aucun fichier de métriques généré"
    assert result.stdout.strip(), "Sortie de script vide"
