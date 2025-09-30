import argparse
import sys
import types

import pytest


matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
pyplot_stub.rcParams = {}
pyplot_stub.rcdefaults = lambda: None
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules["matplotlib.pyplot"] = pyplot_stub

from scripts.mne3sd.common import add_worker_argument, resolve_worker_count


@pytest.mark.parametrize("value, expected", [("3", 3), ("auto", "auto")])
def test_add_worker_argument_accepts_int_and_auto(value, expected):
    parser = argparse.ArgumentParser()
    add_worker_argument(parser)
    args = parser.parse_args(["--workers", value])
    assert args.workers == expected


def test_add_worker_argument_rejects_invalid_values():
    parser = argparse.ArgumentParser()
    add_worker_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--workers", "0"])


def test_resolve_worker_count_limits_to_tasks(monkeypatch):
    monkeypatch.setattr("scripts.mne3sd.common.os.cpu_count", lambda: 8)
    assert resolve_worker_count("auto", 3) == 3
    assert resolve_worker_count("auto", 12) == 8


@pytest.mark.parametrize("workers, tasks, expected", [(4, 2, 2), (4, 6, 4)])
def test_resolve_worker_count_with_explicit_integer(workers, tasks, expected):
    assert resolve_worker_count(workers, tasks) == expected


def test_resolve_worker_count_without_tasks_returns_zero():
    assert resolve_worker_count("auto", 0) == 0
