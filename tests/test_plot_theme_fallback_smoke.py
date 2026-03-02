from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def test_plot_step1_results_generates_png_without_article_c(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    algo_dir = results_dir / "adr"
    algo_dir.mkdir(parents=True)

    csv_path = algo_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "num_nodes",
                "packet_interval_s",
                "snir_state",
                "snir_mean",
                "PDR",
                "DER",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "algorithm": "adr",
                "num_nodes": "100",
                "packet_interval_s": "60",
                "snir_state": "snir_on",
                "snir_mean": "12.0",
                "PDR": "0.95",
                "DER": "0.85",
            }
        )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_step1_results.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--no-compare-snir",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "article_c indisponible" in completed.stderr
    png_files = list(figures_dir.rglob("*.png"))
    assert png_files, "Aucun PNG généré en mode fallback sans article_c."
