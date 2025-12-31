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
import pandas as pd

from qos_cli import lfs_plots
from qos_cli import lfs_metrics
from qos_cli.lfs_metrics import MethodScenarioMetrics

matplotlib.use("Agg")


def test_plot_snir_cdf_snr_fallback_label(tmp_path: Path, monkeypatch) -> None:
    scenario = "Scenario-SNR"
    output_dir = tmp_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"snr_dB": [-1.2, 0.3, 2.1, 4.0]})
    snir_cdf = lfs_metrics.compute_snir_cdf(df)
    snr_cdf = lfs_metrics.compute_snr_cdf(df)

    metric = MethodScenarioMetrics(
        method="AlgoA",
        scenario=scenario,
        use_snir=True,
        snir_state="snir_on",
        delivered=4,
        attempted=4,
        cluster_pdr={},
        cluster_targets={},
        pdr_gap_by_cluster={},
        pdr_global=1.0,
        der_global=0.0,
        collisions=0,
        collision_rate=0.0,
        snir_cdf=snir_cdf,
        snr_cdf=snr_cdf,
        energy_j=None,
        energy_per_delivery=None,
        energy_per_attempt=None,
        jain_index=None,
        min_sf_share=None,
        loss_rate=None,
    )

    metrics_by_method = {"AlgoA": {scenario: metric}}

    captured_legends: dict[str, list[str]] = {}
    original_savefig = Figure.savefig

    def _spy_savefig(self: Figure, fname: str | Path, *args, **kwargs):  # type: ignore[override]
        handles, labels = (self.axes[0].get_legend_handles_labels() if self.axes else ([], []))
        captured_legends[str(Path(fname))] = [label for label in labels if label]
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy_savefig)

    generated = lfs_plots.plot_snir_cdf(metrics_by_method, [scenario], output_dir)
    assert generated

    mixed_path = output_dir / f"snr_cdf_{lfs_plots.sanitize_filename(scenario)}_snir-mixed.png"
    labels = captured_legends.get(str(mixed_path), [])
    assert labels, "La légende doit contenir la courbe de fallback."
    assert any("SNR" in label for label in labels)
    assert not snir_cdf
