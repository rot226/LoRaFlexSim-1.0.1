"""Tests for the mobility range sweep CLI helpers."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def range_module(monkeypatch: pytest.MonkeyPatch):
    """Load the range sweep module with lightweight dependency stubs."""

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)

    mne3sd_pkg = types.ModuleType("scripts.mne3sd")
    mne3sd_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.mne3sd", mne3sd_pkg)

    common_stub = types.ModuleType("scripts.mne3sd.common")

    def _identity(value):
        return value

    common_stub.add_execution_profile_argument = lambda *args, **kwargs: None
    common_stub.add_worker_argument = lambda *args, **kwargs: None
    common_stub.filter_completed_tasks = lambda *args, **kwargs: []
    common_stub.resolve_execution_profile = _identity
    common_stub.resolve_worker_count = lambda *args, **kwargs: 1
    common_stub.summarise_metrics = lambda *args, **kwargs: []
    common_stub.write_csv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "scripts.mne3sd.common", common_stub)

    loraflexsim_pkg = types.ModuleType("loraflexsim")
    loraflexsim_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "loraflexsim", loraflexsim_pkg)

    launcher_stub = types.ModuleType("loraflexsim.launcher")
    launcher_stub.RandomWaypoint = object
    launcher_stub.Simulator = object
    launcher_stub.SmoothMobility = object
    monkeypatch.setitem(sys.modules, "loraflexsim.launcher", launcher_stub)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mne3sd"
        / "article_b"
        / "scenarios"
        / "run_mobility_range_sweep.py"
    )

    spec = importlib.util.spec_from_file_location("range_module_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_range_list_rejects_values_above_maximum(range_module) -> None:
    """Values above MAX_RANGE_KM must trigger an ArgumentTypeError."""

    above_max = range_module.MAX_RANGE_KM + 0.1

    with pytest.raises(argparse.ArgumentTypeError):
        range_module.parse_range_list([str(above_max)])
