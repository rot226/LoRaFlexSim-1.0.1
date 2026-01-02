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
sys.modules.pop("numpy.exceptions", None)

numpy = importlib.import_module("numpy")
sys.modules["numpy"] = numpy
sys.modules["numpy.random"] = numpy.random
sys.modules["numpy.linalg"] = numpy.linalg

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")


def _load_plot_module() -> object:
    root_dir = Path(__file__).resolve().parents[2]
    module_path = root_dir / "experiments" / "ucb1" / "plots" / "plot_regret.py"
    spec = importlib.util.spec_from_file_location("plot_regret", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger plot_regret.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_regret_includes_snir_curves(tmp_path: Path, monkeypatch) -> None:
    plot_regret = _load_plot_module()
    csv_path = tmp_path / "ucb1_regret.csv"
    output_path = tmp_path / "ucb1_regret.png"

    rows = [
        {"cluster": 1, "packet_interval_s": 600.0, "window_index": 0, "window_start_s": 0.0, "regret_cumulative": 0.1, "snir_state": "snir_off"},
        {"cluster": 1, "packet_interval_s": 600.0, "window_index": 1, "window_start_s": 600.0, "regret_cumulative": 0.2, "snir_state": "snir_off"},
        {"cluster": 1, "packet_interval_s": 600.0, "window_index": 0, "window_start_s": 0.0, "regret_cumulative": 0.05, "snir_state": "snir_on"},
        {"cluster": 1, "packet_interval_s": 600.0, "window_index": 1, "window_start_s": 600.0, "regret_cumulative": 0.12, "snir_state": "snir_on"},
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

    generated = plot_regret.plot_regret(
        csv_path=csv_path,
        output_path=output_path,
        packet_intervals=[],
    )

    assert generated.is_file()
    assert generated.stat().st_size > 0
    assert any(plot_regret.SNIR_LABELS["snir_on"] in label for label in captured_labels)
    assert any(plot_regret.SNIR_LABELS["snir_off"] in label for label in captured_labels)
