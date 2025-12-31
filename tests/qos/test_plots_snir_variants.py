from __future__ import annotations

import importlib
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

from qos_cli import lfs_plots
from qos_cli.lfs_metrics import MethodScenarioMetrics

matplotlib.use("Agg")


def _make_metrics(method: str, scenario: str, *, snir_state: str, pdr: float) -> MethodScenarioMetrics:
    delivered = int(round(pdr * 10))
    return MethodScenarioMetrics(
        method=method,
        scenario=scenario,
        use_snir=snir_state == "snir_on",
        snir_state=snir_state,
        delivered=delivered,
        attempted=10,
        cluster_pdr={},
        cluster_targets={},
        pdr_gap_by_cluster={},
        pdr_global=pdr,
        der_global=1.0 - pdr,
        collisions=0,
        collision_rate=0.0,
        snir_cdf=[],
        snr_cdf=[],
        energy_j=None,
        energy_per_delivery=None,
        energy_per_attempt=None,
        jain_index=None,
        min_sf_share=None,
        loss_rate=None,
    )


def test_plot_pdr_generates_snir_variants(tmp_path: Path, monkeypatch) -> None:
    scenario = "Scenario-PDR"
    output_dir = tmp_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_method = {
        "AlgoA": {scenario: _make_metrics("AlgoA", scenario, snir_state="snir_on", pdr=0.8)},
        "AlgoB": {scenario: _make_metrics("AlgoB", scenario, snir_state="snir_off", pdr=0.4)},
    }

    captured_legends: dict[str, tuple[list[str], list[str]]] = {}
    original_savefig = Figure.savefig

    def _spy_savefig(self: Figure, fname: str | Path, *args, **kwargs):  # type: ignore[override]
        handles, labels = (self.axes[0].get_legend_handles_labels() if self.axes else ([], []))
        colors = [getattr(handle, "get_color", lambda: None)() for handle in handles]
        captured_legends[str(Path(fname))] = (labels, colors)
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy_savefig)

    generated_paths = lfs_plots.plot_pdr(metrics_by_method, [scenario], output_dir)

    suffixes = ("_snir-on.png", "_snir-off.png", "_snir-mixed.png")
    expected_paths = {output_dir / f"pdr_global_vs_scenarios{suffix}" for suffix in suffixes}
    assert set(generated_paths) == expected_paths

    for path in expected_paths:
        assert path.is_file()
        assert path.stat().st_size > 0

    mixed_path = str(output_dir / "pdr_global_vs_scenarios_snir-mixed.png")
    labels, colors = captured_legends.get(mixed_path, ([], []))
    assert labels == [
        f"AlgoA ({lfs_plots.SNIR_STATE_LABELS['snir_on']})",
        f"AlgoB ({lfs_plots.SNIR_STATE_LABELS['snir_off']})",
    ]
    assert colors == [
        lfs_plots.SNIR_STATE_COLORS["snir_on"],
        lfs_plots.SNIR_STATE_COLORS["snir_off"],
    ]
