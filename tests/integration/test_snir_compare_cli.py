from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"CSV vide : {path}"
    return rows


def _metric_stats(rows: list[dict[str, str]], key: str, cast) -> dict[str, float]:
    values = [cast(row[key]) for row in rows]
    avg = mean(values)
    if len(values) > 1:
        std = stdev(values)
        ci95 = 1.96 * std / math.sqrt(len(values))
    else:
        std = 0.0
        ci95 = 0.0
    return {"mean": avg, "stdev": std, "ci95": ci95}


def test_compare_generates_differences(tmp_path: Path) -> None:
    outdir = tmp_path / "snir_compare"
    algorithms = ["interference_only", "snir_interference"]
    reps = 3
    script = Path(__file__).resolve().parents[2] / "experiments" / "snir_stage1_compare" / "scenarios" / "run_compare_stage1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--algorithms",
            ",".join(algorithms),
            "--profiles",
            "flora_full",
            "--nodes",
            "20",
            "--intervals",
            "1.0",
            "--reps",
            str(reps),
            "--jobs",
            "1",
            "--seed",
            "7",
            "--outdir",
            str(outdir),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    expected_files = {f"{algo}_compare.csv" for algo in algorithms}
    produced = {path.name for path in outdir.glob("*.csv")}
    assert produced == expected_files, f"Fichiers générés inattendus : {produced}"

    baseline_rows = _read_rows(outdir / "interference_only_compare.csv")
    snir_rows = _read_rows(outdir / "snir_interference_compare.csv")
    assert len(baseline_rows) == reps
    assert len(snir_rows) == reps

    baseline_der = _metric_stats(baseline_rows, "der", float)
    snir_der = _metric_stats(snir_rows, "der", float)
    baseline_collisions = _metric_stats(baseline_rows, "collisions", int)
    snir_collisions = _metric_stats(snir_rows, "collisions", int)
    baseline_snir = _metric_stats(baseline_rows, "snir_mean", float)
    snir_snir = _metric_stats(snir_rows, "snir_mean", float)

    assert all(stats["stdev"] > 0 for stats in (baseline_der, snir_der)), "Variance DER absente"
    assert snir_collisions["stdev"] > 0, "Variance des collisions SNIR absente"

    ci_penalty = (baseline_der["ci95"] + snir_der["ci95"]) + (baseline_collisions["ci95"] + snir_collisions["ci95"]) + (baseline_snir["ci95"] + snir_snir["ci95"])

    der_delta = abs(baseline_der["mean"] - snir_der["mean"]) - (baseline_der["ci95"] + snir_der["ci95"])
    collisions_delta = abs(baseline_collisions["mean"] - snir_collisions["mean"]) - (baseline_collisions["ci95"] + snir_collisions["ci95"])
    snir_delta = abs(baseline_snir["mean"] - snir_snir["mean"]) - (baseline_snir["ci95"] + snir_snir["ci95"])

    assert der_delta >= 0.02, f"Delta DER trop faible ({der_delta:.4f})"
    assert collisions_delta >= 10, f"Delta collisions trop faible ({collisions_delta:.2f})"
    assert snir_delta >= 5.0, f"Delta SNIR moyen trop faible ({snir_delta:.2f})"
    assert ci_penalty >= 0, "Calcul d'intervalle de confiance incohérent"
    assert any(outdir.iterdir()), "Aucun fichier de métriques généré"
    assert result.stdout.strip(), "Sortie de script vide"
