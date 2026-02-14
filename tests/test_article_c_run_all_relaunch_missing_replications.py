from __future__ import annotations

import argparse
from pathlib import Path

from article_c import run_all
from article_c.common.utils import replication_ids


def test_relaunch_missing_replication_creates_expected_nested_directory(tmp_path):
    results_dir = tmp_path / "results"
    size = 80
    size_dir = results_dir / "by_size" / f"size_{size}"

    for rep in (0, 2):
        rep_dir = size_dir / f"rep_{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "aggregated_results.csv").write_text(
            "network_size,success_rate_mean\n80,0.5\n",
            encoding="utf-8",
        )

    missing_before = run_all._missing_replications_by_size(
        results_dir,
        [size],
        replications_total=3,
    )
    assert missing_before == {size: [1]}

    base_args = argparse.Namespace(
        network_sizes=[size],
        replications=3,
        seeds_base=7,
        flat_output=True,
        reset_status=True,
    )

    invocations: list[argparse.Namespace] = []

    def _build_args(args: argparse.Namespace) -> argparse.Namespace:
        return args

    def _fake_runner(args: argparse.Namespace) -> None:
        invocations.append(args)
        assert args.network_sizes == [size]
        assert args.replications == 2
        assert args.seeds_base == 7
        assert args.flat_output is False
        target_rep = args.replications - 1
        rep_dir = size_dir / f"rep_{target_rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "aggregated_results.csv").write_text(
            "network_size,success_rate_mean\n80,0.75\n",
            encoding="utf-8",
        )

    relaunched = run_all._relaunch_missing_replications(
        step_label="Step2",
        results_dir=results_dir,
        missing_by_size=missing_before,
        build_args=_build_args,
        runner=_fake_runner,
        base_args=base_args,
    )

    assert relaunched == [(size, 1)]
    assert len(invocations) == 1
    assert (size_dir / "rep_1").is_dir()

    expected_rep_dirs = {f"rep_{rep}" for rep in replication_ids(3)}
    actual_rep_dirs = {path.name for path in size_dir.glob("rep_*") if path.is_dir()}
    assert actual_rep_dirs == expected_rep_dirs

    missing_after = run_all._missing_replications_by_size(
        results_dir,
        [size],
        replications_total=3,
    )
    assert missing_after == {}
