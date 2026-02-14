from __future__ import annotations

from article_c import run_all


class _Called(RuntimeError):
    """Signal interne pour arrêter run_all après le démarrage effectif d'une étape."""


def _patch_smoke_startup(monkeypatch, *, expect_step: str) -> dict[str, int]:
    calls = {"step1": 0, "step2": 0}

    def _fake_step1(_argv):
        calls["step1"] += 1
        if expect_step == "step1":
            raise _Called("step1 started")
        return None

    def _fake_step2(_argv):
        calls["step2"] += 1
        if expect_step == "step2":
            raise _Called("step2 started")
        return None

    monkeypatch.setattr(run_all, "_enforce_article_c_branch", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_global_aggregation_artifacts", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_no_global_writes_during_simulation", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_output_layout_compliant", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_aggregation_contract_consistent", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes_nested", lambda *_: None)
    monkeypatch.setattr(run_all, "aggregate_results_by_size", lambda *_args, **_kwargs: {"global_row_count": 0})
    monkeypatch.setattr(run_all, "validate_results", lambda *_: 0)
    monkeypatch.setattr(run_all, "run_step1", _fake_step1)
    monkeypatch.setattr(run_all, "run_step2", _fake_step2)

    return calls


def test_smoke_skip_step2_starts_before_any_argparse_attribute_error(monkeypatch):
    calls = _patch_smoke_startup(monkeypatch, expect_step="step1")

    argv = ["--network-sizes", "80", "--replications", "1", "--skip-step2"]

    try:
        run_all.main(argv)
    except _Called:
        pass
    except AttributeError as exc:  # pragma: no cover - message explicite pour diagnostic
        raise AssertionError(f"Argparse attribute error détectée avant exécution effective: {exc}") from exc

    assert calls["step1"] == 1
    assert calls["step2"] == 0


def test_smoke_skip_step1_starts_before_any_argparse_attribute_error(monkeypatch):
    calls = _patch_smoke_startup(monkeypatch, expect_step="step2")

    argv = ["--network-sizes", "80", "--replications", "1", "--skip-step1"]

    try:
        run_all.main(argv)
    except _Called:
        pass
    except AttributeError as exc:  # pragma: no cover - message explicite pour diagnostic
        raise AssertionError(f"Argparse attribute error détectée avant exécution effective: {exc}") from exc

    assert calls["step1"] == 0
    assert calls["step2"] == 1


def test_run_all_skip_step2_no_forbidden_global_write_exception(monkeypatch):
    checks = {"calls": 0}
    step1_kwargs = []

    def _fake_step1(_argv, **kwargs):
        step1_kwargs.append(kwargs)
        return None

    def _fake_assert_no_global(*_args, **_kwargs):
        checks["calls"] += 1

    monkeypatch.setattr(run_all, "_enforce_article_c_branch", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_global_aggregation_artifacts", lambda *_: None)
    monkeypatch.setattr(run_all, "_find_first_missing_rep", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes_nested", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_output_layout_compliant", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_aggregation_contract_consistent", lambda *_: None)
    monkeypatch.setattr(run_all, "_missing_replications_by_size", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_all, "aggregate_results_by_size", lambda *_args, **_kwargs: {"size_count": 1, "size_row_count": 1, "global_row_count": 1})
    monkeypatch.setattr(run_all, "validate_results", lambda *_: 0)
    monkeypatch.setattr(run_all, "_build_step2_quality_summary", lambda *_: {"simulation_quality": "ok", "reasons": []})
    monkeypatch.setattr(run_all, "_count_failed_runs", lambda *_: 0)
    monkeypatch.setattr(run_all, "_write_campaign_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "_log_existing_key_csv_paths", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_done_flag", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_no_global_writes_during_simulation", _fake_assert_no_global)
    monkeypatch.setattr(run_all, "run_step1", _fake_step1)
    monkeypatch.setattr(run_all, "run_step2", lambda *_args, **_kwargs: None)

    run_all.main(["--network-sizes", "80", "--replications", "1", "--skip-step2"])

    assert checks["calls"] == 1
    assert step1_kwargs == [{"write_global_aggregated": False}]


def test_run_all_skip_step1_no_forbidden_global_write_exception(monkeypatch):
    checks = {"calls": 0}
    step2_kwargs = []

    def _fake_step2(_argv, **kwargs):
        step2_kwargs.append(kwargs)
        return None

    def _fake_assert_no_global(*_args, **_kwargs):
        checks["calls"] += 1

    monkeypatch.setattr(run_all, "_enforce_article_c_branch", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_global_aggregation_artifacts", lambda *_: None)
    monkeypatch.setattr(run_all, "_find_first_missing_rep", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(run_all, "_assert_cumulative_sizes_nested", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_output_layout_compliant", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_aggregation_contract_consistent", lambda *_: None)
    monkeypatch.setattr(run_all, "_missing_replications_by_size", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_all, "aggregate_results_by_size", lambda *_args, **_kwargs: {"size_count": 1, "size_row_count": 1, "global_row_count": 1})
    monkeypatch.setattr(run_all, "validate_results", lambda *_: 0)
    monkeypatch.setattr(run_all, "_build_step2_quality_summary", lambda *_: {"simulation_quality": "ok", "reasons": []})
    monkeypatch.setattr(run_all, "_count_failed_runs", lambda *_: 0)
    monkeypatch.setattr(run_all, "_write_campaign_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "_log_existing_key_csv_paths", lambda *_: None)
    monkeypatch.setattr(run_all, "_remove_done_flag", lambda *_: None)
    monkeypatch.setattr(run_all, "_assert_no_global_writes_during_simulation", _fake_assert_no_global)
    monkeypatch.setattr(run_all, "run_step1", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_all, "run_step2", _fake_step2)

    run_all.main(["--network-sizes", "80", "--replications", "1", "--skip-step1"])

    assert checks["calls"] == 1
    assert step2_kwargs == [{"write_global_aggregated": False}]
