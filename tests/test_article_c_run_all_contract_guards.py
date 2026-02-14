from __future__ import annotations

import csv
from pathlib import Path

from article_c import run_all
from article_c.common.csv_io import aggregate_results_by_size


def _write_rep_csv(rep_dir: Path, network_size: int) -> None:
    rep_dir.mkdir(parents=True, exist_ok=True)
    with (rep_dir / "aggregated_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["network_size", "algo", "snir_mode", "cluster", "pdr_mean"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "network_size": network_size,
                "algo": "adr",
                "snir_mode": "snir_on",
                "cluster": "all",
                "pdr_mean": "0.90",
            }
        )


def test_run_all_recovery_cycle_triggers_targeted_relaunch(monkeypatch) -> None:
    relaunch_calls: list[tuple[str, dict[int, list[int]]]] = []

    monkeypatch.setattr(run_all, "_enforce_article_c_branch", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_global_aggregation_artifacts", lambda *_: None)
    monkeypatch.setattr(run_all, "_find_first_missing_rep", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(run_all, "_write_campaign_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "_remove_done_flag", lambda *_: None)
    monkeypatch.setattr(run_all, "_read_campaign_state", lambda *_: None)
    monkeypatch.setattr(run_all, "_cleanup_size_directory", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_no_global_writes_during_simulation", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes_nested", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_output_layout_compliant", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_aggregation_contract_consistent", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_required_aggregates_present", lambda *_: None)
    monkeypatch.setattr(run_all, "_log_existing_key_csv_paths", lambda *_: None)
    monkeypatch.setattr(run_all, "_count_failed_runs", lambda *_: 0)
    monkeypatch.setattr(run_all, "_build_step2_quality_summary", lambda *_: {"simulation_quality": "ok", "reasons": []})
    monkeypatch.setattr(run_all, "aggregate_results_by_size", lambda *_args, **_kwargs: {"global_row_count": 1})
    monkeypatch.setattr(run_all, "validate_results", lambda *_: 0)
    monkeypatch.setattr(run_all, "run_step1", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "run_step2", lambda *_args, **_kwargs: None)

    missing_answers = iter([{80: [1]}, {80: [0]}])
    monkeypatch.setattr(
        run_all,
        "_missing_replications_by_size",
        lambda *_args, **_kwargs: next(missing_answers),
    )

    def _fake_relaunch(*, step_label: str, missing_by_size: dict[int, list[int]], **_kwargs):
        relaunch_calls.append((step_label, missing_by_size))
        return [(size, rep) for size, reps in sorted(missing_by_size.items()) for rep in reps]

    monkeypatch.setattr(run_all, "_relaunch_missing_replications", _fake_relaunch)

    run_all.main(["--network-sizes", "80", "--replications", "2"])

    assert relaunch_calls == [("Step1", {80: [1]}), ("Step2", {80: [0]})]


def test_step1_step2_layout_and_final_aggregates_contract(tmp_path: Path) -> None:
    expected_sizes = [80]
    replications_total = 2

    for step_label in ("Step1", "Step2"):
        results_dir = tmp_path / step_label.lower() / "results"
        for rep in range(replications_total):
            _write_rep_csv(results_dir / "by_size" / "size_80" / f"rep_{rep}", 80)

        (results_dir / "by_size" / "size_80" / "aggregated_results.csv").write_text(
            "network_size,algo,snir_mode,cluster,pdr_mean\n80,adr,snir_on,all,0.90\n",
            encoding="utf-8",
        )

        run_all._assert_output_layout_compliant(
            results_dir,
            expected_sizes=expected_sizes,
            replications_total=replications_total,
            step_label=step_label,
        )

        run_all._assert_aggregation_contract_consistent(
            results_dir,
            expected_sizes=expected_sizes,
            step_label=step_label,
        )

        stats = aggregate_results_by_size(results_dir, write_global_aggregated=True)
        assert stats["global_row_count"] > 0
        assert (results_dir / "aggregates" / "aggregated_results.csv").exists()
        assert (results_dir / "by_size" / "size_80" / "aggregated_results.csv").exists()

