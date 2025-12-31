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
from matplotlib.axes import Axes
import numpy as np

from qos_cli import lfs_plots
from qos_cli.lfs_metrics import MethodScenarioMetrics

matplotlib.use("Agg")


def _make_metric(
    *,
    method: str,
    scenario: str,
    snir_state: str,
    delivered: int,
    attempted: int,
    snir_mean: float | None = None,
    snir_ci_low: float | None = None,
    snir_ci_high: float | None = None,
) -> MethodScenarioMetrics:
    ratio = delivered / attempted if attempted else 0.0
    return MethodScenarioMetrics(
        method=method,
        scenario=scenario,
        use_snir=snir_state == "snir_on",
        snir_state=snir_state,
        delivered=delivered,
        attempted=attempted,
        cluster_pdr={},
        cluster_targets={},
        pdr_gap_by_cluster={},
        pdr_global=ratio,
        der_global=ratio,
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
        snir_mean=snir_mean,
        snir_ci_low=snir_ci_low,
        snir_ci_high=snir_ci_high,
    )


def _has_ci_call(calls: list[tuple[tuple[object, ...], dict]], expected_low: float, expected_high: float) -> bool:
    for args, _kwargs in calls:
        if len(args) < 3:
            continue
        y1 = np.asarray(args[1], dtype=float)
        y2 = np.asarray(args[2], dtype=float)
        if y1.size == 0 or y2.size == 0:
            continue
        if np.isclose(y1[0], expected_low) and np.isclose(y2[0], expected_high):
            return True
    return False


def test_plot_pdr_der_include_ic95(tmp_path: Path, monkeypatch) -> None:
    scenario = "Scenario-CI"
    metric = _make_metric(
        method="AlgoA",
        scenario=scenario,
        snir_state="snir_on",
        delivered=8,
        attempted=10,
    )
    metrics_by_method = {"AlgoA": {scenario: metric}}

    calls: list[tuple[tuple[object, ...], dict]] = []
    original_fill_between = Axes.fill_between

    def _spy_fill_between(self: Axes, *args, **kwargs):  # type: ignore[override]
        calls.append((args, kwargs))
        return original_fill_between(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", _spy_fill_between)

    expected_low, expected_high = lfs_plots._ratio_confidence(8, 10)
    lfs_plots.plot_pdr(metrics_by_method, [scenario], tmp_path)
    assert _has_ci_call(calls, expected_low, expected_high)

    calls.clear()
    lfs_plots.plot_der(metrics_by_method, [scenario], tmp_path)
    assert _has_ci_call(calls, expected_low, expected_high)


def test_plot_snir_mean_has_variants_and_ic95(tmp_path: Path, monkeypatch) -> None:
    scenario = "Scenario-SNIR"
    metrics_by_method = {
        "AlgoA": {
            scenario: _make_metric(
                method="AlgoA",
                scenario=scenario,
                snir_state="snir_on",
                delivered=9,
                attempted=10,
                snir_mean=5.0,
                snir_ci_low=4.5,
                snir_ci_high=5.5,
            )
        },
        "AlgoB": {
            scenario: _make_metric(
                method="AlgoB",
                scenario=scenario,
                snir_state="snir_off",
                delivered=7,
                attempted=10,
                snir_mean=2.0,
                snir_ci_low=1.0,
                snir_ci_high=3.0,
            )
        },
    }

    calls: list[tuple[tuple[object, ...], dict]] = []
    original_fill_between = Axes.fill_between

    def _spy_fill_between(self: Axes, *args, **kwargs):  # type: ignore[override]
        calls.append((args, kwargs))
        return original_fill_between(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", _spy_fill_between)

    generated = lfs_plots.plot_snir_mean(metrics_by_method, [scenario], tmp_path)
    expected_suffixes = ("_snir-on.png", "_snir-off.png", "_snir-mixed.png")
    expected_paths = {tmp_path / f"snir_mean_vs_scenarios{suffix}" for suffix in expected_suffixes}
    assert set(generated) == expected_paths
    for path in expected_paths:
        assert path.is_file()
        assert path.stat().st_size > 0

    assert any(call for call in calls), "Les IC95 doivent être tracés via fill_between."
