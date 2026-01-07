from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
from pathlib import Path

STUBS_DIR = Path(__file__).resolve().parents[1] / "stubs"
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))

sys.modules.pop("numpy", None)
sys.modules.pop("numpy.random", None)
sys.modules.pop("numpy.linalg", None)
sys.modules.pop("numpy.ma", None)
sys.modules.pop("numpy.exceptions", None)

numpy = importlib.import_module("numpy")
numpy_random = importlib.import_module("numpy.random")
numpy_linalg = importlib.import_module("numpy.linalg")
numpy_ma = importlib.import_module("numpy.ma")
setattr(numpy, "random", numpy_random)
setattr(numpy, "linalg", numpy_linalg)
setattr(numpy, "ma", numpy_ma)
sys.modules["numpy"] = numpy
sys.modules["numpy.random"] = numpy_random
sys.modules["numpy.linalg"] = numpy_linalg
sys.modules["numpy.ma"] = numpy_ma

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")


def _load_plot_module() -> object:
    root_dir = Path(__file__).resolve().parents[2]
    module_path = root_dir / "experiments" / "ucb1" / "plots" / "plot_mab_timeseries.py"
    spec = importlib.util.spec_from_file_location("plot_mab_timeseries", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger plot_mab_timeseries.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_mab_timeseries_has_labels(tmp_path: Path, monkeypatch) -> None:
    plot_module = _load_plot_module()
    csv_path = tmp_path / "ucb1_timeseries.csv"
    output_dir = tmp_path / "plots"

    rows = [
        {
            "cluster": 1,
            "packet_interval_s": 600.0,
            "window_index": 0,
            "window_start_s": 0.0,
            "reward_window_mean": 0.8,
            "der_window": 0.5,
            "snir_window_mean": 7.5,
            "energy_window_mean": 0.12,
            "snir_state": "snir_on",
        },
        {
            "cluster": 1,
            "packet_interval_s": 600.0,
            "window_index": 1,
            "window_start_s": 600.0,
            "reward_window_mean": 0.7,
            "der_window": 0.6,
            "snir_window_mean": 7.1,
            "energy_window_mean": 0.11,
            "snir_state": "snir_on",
        },
    ]

    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    captured_labels: list[str] = []
    original_savefig = Figure.savefig

    def _spy_savefig(self: Figure, fname: str | Path, *args, **kwargs):  # type: ignore[override]
        labels = self.axes[0].get_legend_handles_labels()[1] if self.axes else []
        captured_labels.extend(labels)
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy_savefig)

    plot_module.run_plots(
        csv_path=csv_path,
        output_dir=output_dir,
        packet_intervals=[],
    )

    assert output_dir.is_dir()
    assert captured_labels
