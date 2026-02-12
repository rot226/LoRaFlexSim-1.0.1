from __future__ import annotations

from article_c import run_all


def _patch_run_all_startup(monkeypatch):
    monkeypatch.setattr(run_all, "_enforce_article_c_branch", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_global_aggregation_artifacts", lambda *_: None)
    monkeypatch.setattr(run_all, "run_step1", lambda *_: None)
    monkeypatch.setattr(run_all, "run_step2", lambda *_: None)
    monkeypatch.setattr(run_all, "aggregate_results_by_size", lambda *_args, **_kwargs: {"global_row_count": 0})
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes_nested", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_output_layout_compliant", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes", lambda *_: None)
    monkeypatch.setattr(run_all, "validate_results", lambda *_: 0)



def test_run_all_skip_step2_never_crashes_on_missing_optional_attrs(monkeypatch):
    _patch_run_all_startup(monkeypatch)
    run_all.main(["--skip-step2"])



def test_run_all_skip_step1_never_crashes_on_missing_optional_attrs(monkeypatch):
    _patch_run_all_startup(monkeypatch)
    run_all.main(["--skip-step1"])
