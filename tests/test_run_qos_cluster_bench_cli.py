from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_qos_cluster_bench
from loraflexsim.scenarios.qos_cluster_presets import get_preset


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


def test_main_uses_preset_quick(tmp_results_dir: Path):
    captured_kwargs = {}

    def fake_runner(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        tmp_results_dir.mkdir(parents=True, exist_ok=True)
        return {"report_path": "docs/report.md", "summary_path": "results/summary.json"}

    summary = run_qos_cluster_bench.main(
        ["--preset", "quick", "--quiet", "--output-dir", str(tmp_results_dir)], runner=fake_runner
    )
    preset = get_preset("quick")
    assert summary["report_path"] == "docs/report.md"
    assert tuple(captured_kwargs["node_counts"]) == tuple(preset.node_counts)
    assert tuple(captured_kwargs["tx_periods"]) == tuple(preset.tx_periods)


def test_list_presets_short_circuit(capsys):
    summary = run_qos_cluster_bench.main(["--list-presets"], runner=lambda **_: {})
    captured = capsys.readouterr()
    assert "Préréglages disponibles" in captured.out
    assert summary == {}
