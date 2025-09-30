import argparse
import importlib
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


def test_fast_profile_limits_mobility_speed_tasks(monkeypatch, tmp_path):
    module = importlib.import_module(
        "scripts.mne3sd.article_b.scenarios.run_mobility_speed_sweep"
    )

    recorded: list[dict[str, object]] = []

    def fake_run(task: dict[str, object]) -> dict[str, object]:
        recorded.append(task)
        return {
            "model": task["model"],
            "speed_profile": task["speed_profile"],
            "speed_min_mps": task["speed_min_mps"],
            "speed_max_mps": task["speed_max_mps"],
            "replicate": task["replicate"],
            "pdr": 0.0,
            "avg_delay_s": 0.0,
            "jitter_s": 0.0,
            "energy_per_node_J": 0.0,
        }

    monkeypatch.setattr(module, "_run_speed_replicate", fake_run)
    monkeypatch.setattr(module, "RESULTS_PATH", tmp_path / "mobility_speed_metrics.csv")
    monkeypatch.setattr(module, "summarise_metrics", lambda *_, **__: [])
    monkeypatch.setattr(module, "write_csv", lambda *_, **__: None)

    def run_with_args(*extra_args: str) -> list[dict[str, object]]:
        recorded.clear()
        sys.argv = ["prog", *extra_args]
        module.main()
        return list(recorded)

    default_tasks = run_with_args(
        "--profile",
        "full",
        "--replicates",
        "5",
        "--nodes",
        "150",
        "--packets",
        "60",
        "--workers",
        "1",
    )

    fast_tasks = run_with_args(
        "--profile",
        "fast",
        "--replicates",
        "5",
        "--nodes",
        "150",
        "--packets",
        "60",
        "--workers",
        "1",
    )

    assert len(fast_tasks) < len(default_tasks)
    assert {task["speed_profile"] for task in fast_tasks} <= {
        entry[0] for entry in module.FAST_SPEED_PROFILES
    }
    assert {task["replicate"] for task in fast_tasks} == set(
        range(1, module.FAST_REPLICATES + 1)
    )
