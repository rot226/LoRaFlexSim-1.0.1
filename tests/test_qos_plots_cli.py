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
import pandas as pd
import pytest

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


def test_rolling_metrics_respects_window_size() -> None:
    timeseries = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "delivered": [1, 0, 1, 0],
            "snir": [10.0, 0.0, 5.0, 15.0],
        }
    )

    short_window = lfs_plots._rolling_metrics(timeseries, window_size=2, window_mode="packets")
    long_window = lfs_plots._rolling_metrics(timeseries, window_size=3, window_mode="packets")

    assert short_window.iloc[-1]["pdr"] == pytest.approx(0.5)
    assert long_window.iloc[-1]["pdr"] == pytest.approx(1 / 3)

    assert short_window.iloc[-1]["snir"] == pytest.approx(10.0)
    assert long_window.iloc[-1]["snir"] == pytest.approx(20.0 / 3.0)


def test_rolling_plot_mentions_window(monkeypatch, tmp_path: Path) -> None:
    data_per_method = {
        "algo": pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "pdr": [0.25, 0.5, 0.75, 0.5],
                "pdr_low": [0.2, 0.3, 0.6, 0.3],
                "pdr_high": [0.3, 0.7, 0.9, 0.7],
            }
        )
    }

    captured_figs: list[object] = []
    captured_legend: dict[str, str] = {}

    def _capture_close(fig):
        captured_figs.append(fig)

    from matplotlib.legend import Legend

    original_set_title = Legend.set_title

    def _capture_set_title(self, title=None, *args, **kwargs):  # type: ignore[override]
        if title is not None:
            captured_legend["title"] = title
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(Legend, "set_title", _capture_set_title)
    monkeypatch.setattr(lfs_plots.plt, "close", _capture_close)

    output = lfs_plots._plot_rolling_metric_for_scenario(
        "pdr",
        data_per_method,
        "Scenario-Beta",
        tmp_path,
        window_size=3,
        window_mode="packets",
    )

    assert output is not None and output.exists()
    assert captured_figs, "La figure doit être conservée pour inspection"
    axes = captured_figs[0].axes[0]
    assert "fenêtre 3 paquets" in axes.get_title()
    assert captured_legend.get("title") == "fenêtre 3 paquets"
