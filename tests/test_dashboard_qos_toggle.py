from __future__ import annotations

import csv
import importlib
import sys

import pytest

_ORIGINAL_NUMPY = None
_ORIGINAL_NUMPY_RANDOM = None
_current_numpy = sys.modules.get("numpy")
if _current_numpy is not None and getattr(_current_numpy, "__name__", "") == "numpy_stub":
    _ORIGINAL_NUMPY = _current_numpy
    _ORIGINAL_NUMPY_RANDOM = sys.modules.get("numpy.random")
    sys.modules.pop("numpy", None)
    sys.modules.pop("numpy.random", None)
    real_numpy = importlib.import_module("numpy")
    sys.modules["numpy"] = real_numpy
    sys.modules["numpy.random"] = real_numpy.random

    for module_name in [
        "loraflexsim.launcher.dashboard",
        "loraflexsim.launcher.simulator",
        "loraflexsim.launcher.node",
        "loraflexsim.launcher.channel",
        "loraflexsim.launcher.multichannel",
        "loraflexsim.launcher.gateway",
        "loraflexsim.launcher.smooth_mobility",
        "traffic.rng_manager",
        "traffic.exponential",
    ]:
        sys.modules.pop(module_name, None)


try:
    dashboard = pytest.importorskip("loraflexsim.launcher.dashboard")
except Exception as exc:
    pytest.skip(f"dashboard import failed: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def reset_dashboard_state():
    """Assure un état cohérent du tableau de bord pour chaque test."""
    if _ORIGINAL_NUMPY is not None and getattr(sys.modules.get("numpy"), "__name__", "") == "numpy_stub":
        sys.modules.pop("numpy", None)
        sys.modules.pop("numpy.random", None)
        real_numpy = importlib.import_module("numpy")
        sys.modules["numpy"] = real_numpy
        sys.modules["numpy.random"] = real_numpy.random
    original_add_callback = dashboard.pn.state.add_periodic_callback

    class _DummyCallback:
        def __init__(self, func=None, period=None, timeout=None):
            self.callback = func
            self.period = period
            self.timeout = timeout

        def stop(self):
            pass

    def _fake_add_periodic_callback(callback=None, period=None, timeout=None):
        return _DummyCallback(callback, period, timeout)

    dashboard.pn.state.add_periodic_callback = _fake_add_periodic_callback
    try:
        dashboard._cleanup_callbacks()
        default_algorithm = dashboard.qos_algorithm_select.value
        dashboard.sim = None
        dashboard.paused = False
        dashboard.runs_events.clear()
        dashboard.runs_metrics.clear()
        dashboard.auto_fast_forward = False
        dashboard.pause_prev_disabled = False
        dashboard.qos_toggle.value = False
        dashboard.qos_manager.clusters = []
        dashboard.qos_manager.node_sf_access.clear()
        dashboard.qos_manager.node_clusters.clear()
        yield
    finally:
        dashboard._cleanup_callbacks()
        dashboard.qos_toggle.value = False
        dashboard.qos_algorithm_select.value = default_algorithm
        dashboard.qos_manager.clusters = []
        dashboard.qos_manager.node_sf_access.clear()
        dashboard.qos_manager.node_clusters.clear()
        dashboard.pn.state.add_periodic_callback = original_add_callback


@pytest.fixture(scope="module", autouse=True)
def _restore_numpy_after_module():
    try:
        yield
    finally:
        if _ORIGINAL_NUMPY is not None:
            sys.modules["numpy"] = _ORIGINAL_NUMPY
            if _ORIGINAL_NUMPY_RANDOM is not None:
                sys.modules["numpy.random"] = _ORIGINAL_NUMPY_RANDOM


def _run_single_simulation(*, qos_enabled: bool) -> None:
    dashboard.packets_input.value = 1
    dashboard.num_runs_input.value = 1
    dashboard.qos_toggle.value = qos_enabled
    if qos_enabled:
        dashboard.qos_algorithm_select.value = "MixRA-H"
        dashboard.qos_manager.configure_clusters(
            1,
            proportions=[1.0],
            arrival_rates=[0.1],
            pdr_targets=[0.9],
        )
    dashboard.setup_simulation()
    assert dashboard.sim is not None
    steps = 0
    while dashboard.sim.step():
        steps += 1
        if steps > 5000:
            pytest.fail("Simulation did not finish in time")
    dashboard.on_stop(None)


def test_qos_toggle_preserves_button_states():
    dashboard.packets_input.value = 1
    dashboard.num_runs_input.value = 1
    dashboard.setup_simulation()
    assert dashboard.sim is not None and dashboard.sim.running

    initial_states = {
        "start": dashboard.start_button.disabled,
        "pause": dashboard.pause_button.disabled,
        "fast": dashboard.fast_forward_button.disabled,
        "export": dashboard.export_button.disabled,
    }

    dashboard.qos_toggle.value = True
    assert dashboard.start_button.disabled == initial_states["start"]
    assert dashboard.pause_button.disabled == initial_states["pause"]
    assert dashboard.fast_forward_button.disabled == initial_states["fast"]
    assert dashboard.export_button.disabled == initial_states["export"]

    dashboard.qos_toggle.value = False
    assert dashboard.start_button.disabled == initial_states["start"]
    assert dashboard.pause_button.disabled == initial_states["pause"]
    assert dashboard.fast_forward_button.disabled == initial_states["fast"]
    assert dashboard.export_button.disabled == initial_states["export"]

    while dashboard.sim.step():
        pass
    dashboard.on_stop(None)


def test_qos_toggle_keeps_pause_and_fast_forward_in_sync():
    dashboard.packets_input.value = 1
    dashboard.num_runs_input.value = 1
    dashboard.setup_simulation()
    assert dashboard.sim is not None and dashboard.sim.running

    expected_fast = dashboard.sim.packets_to_send <= 0
    assert dashboard.fast_forward_button.disabled == expected_fast
    assert dashboard.paused is False
    assert getattr(dashboard.sim, "paused", False) is False

    dashboard.qos_toggle.value = True
    assert dashboard.fast_forward_button.disabled == expected_fast
    assert getattr(dashboard.sim, "paused", False) is False

    dashboard.on_pause()
    assert dashboard.paused is True
    assert getattr(dashboard.sim, "paused", False) is True
    assert dashboard.fast_forward_button.disabled is True

    dashboard.qos_toggle.value = False
    assert dashboard.paused is True
    assert getattr(dashboard.sim, "paused", False) is True
    assert dashboard.fast_forward_button.disabled is True

    dashboard.on_pause()
    assert dashboard.paused is False
    assert getattr(dashboard.sim, "paused", False) is False
    assert dashboard.fast_forward_button.disabled == (dashboard.sim.packets_to_send <= 0)

    while dashboard.sim.step():
        pass
    dashboard.on_stop(None)


def test_export_includes_qos_metrics_when_enabled(monkeypatch, tmp_path):
    _run_single_simulation(qos_enabled=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *args, **kwargs: None)

    dashboard.exporter_csv()

    metrics_files = sorted(tmp_path.glob("metrics_*.csv"))
    assert metrics_files, "Metrics file not created"
    with metrics_files[-1].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    assert first_row is not None
    assert "qos_throughput_gini" in first_row
    assert any(key.startswith("qos_cluster_") for key in first_row)
    assert "qos_cluster_throughput_bps.1" in first_row
    assert first_row["qos_cluster_throughput_bps.1"].strip() != ""


def test_export_works_without_qos(monkeypatch, tmp_path):
    _run_single_simulation(qos_enabled=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *args, **kwargs: None)

    dashboard.exporter_csv()

    result_files = sorted(tmp_path.glob("resultats_simulation_*.csv"))
    metrics_files = sorted(tmp_path.glob("metrics_*.csv"))
    assert result_files, "Events export missing"
    assert metrics_files, "Metrics export missing"
    with metrics_files[-1].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    assert first_row is not None
    assert "qos_throughput_gini" in first_row
    assert first_row["qos_throughput_gini"] in ("0", "0.0", "0.0")
    assert not any(key.startswith("qos_cluster_") for key in first_row)
