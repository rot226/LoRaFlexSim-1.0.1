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
