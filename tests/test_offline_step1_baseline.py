from __future__ import annotations

from pathlib import Path

from scripts.run_step1_baseline import expected_baseline_csvs, missing_baseline_csvs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf8")


def test_offline_baseline_expected_csvs(tmp_path: Path) -> None:
    expected = expected_baseline_csvs(tmp_path)
    missing = missing_baseline_csvs(tmp_path)

    assert set(missing) == set(expected)

    for path in expected:
        _touch(path)

    assert missing_baseline_csvs(tmp_path) == []
