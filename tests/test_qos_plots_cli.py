from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

STUBS_DIR = Path(__file__).resolve().parent / "stubs"
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))

sys.modules.pop("numpy", None)
sys.modules.pop("numpy.random", None)
sys.modules.pop("numpy.exceptions", None)

numpy = importlib.import_module("numpy")
sys.modules["numpy"] = numpy
sys.modules["numpy.random"] = numpy.random

import matplotlib

from qos_cli import lfs_plots


matplotlib.use("Agg")


def _write_packets(path: Path, *, snir_state: str, snir_values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"delivered": 1, "snir_state": snir_state, "snir_dB": snir_values[0], "node_id": 1, "cluster": "A"},
        {"delivered": 0, "snir_state": snir_state, "snir_dB": snir_values[1], "node_id": 2, "cluster": "A"},
        {"delivered": 1, "snir_state": snir_state, "snir_dB": snir_values[2], "node_id": 1, "cluster": "B"},
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_generates_all_snir_variants(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    out_dir = tmp_path / "figures"
    scenario = "Scenario-Alpha"

    _write_packets(metrics_root / "algo_on" / scenario / "packets.csv", snir_state="snir_on", snir_values=[7.0, 5.5, 6.2])
    _write_packets(metrics_root / "algo_off" / scenario / "packets.csv", snir_state="snir_off", snir_values=[1.5, 2.0, 0.5])

    lfs_plots.main(["--in", str(metrics_root), "--out", str(out_dir)])

    suffixes = ["_snir-on.png", "_snir-off.png", "_snir-mixed.png"]
    bases = [
        "pdr_global_vs_scenarios",
        "der_global_vs_scenarios",
        f"snir_cdf_{lfs_plots.sanitize_filename(scenario)}",
    ]

    for base in bases:
        for suffix in suffixes:
            path = out_dir / f"{base}{suffix}"
            assert path.is_file(), f"Fichier manquant: {path}"
            assert path.stat().st_size > 0, f"Fichier vide: {path}"
