from __future__ import annotations

import article_c.make_all_plots as make_all_plots


def test_validate_plot_modules_ignores_utils(monkeypatch) -> None:
    monkeypatch.setattr(
        make_all_plots,
        "PLOT_MODULES",
        {"step1": ["article_c.step1.plots.plot_S1"]},
    )

    assert make_all_plots._validate_plot_modules_use_save_figure() == {}


def test_collect_nested_csvs_detects_file_in_by_size(tmp_path) -> None:
    results_dir = tmp_path / "step1" / "results"
    csv_path = results_dir / "by_size" / "size_100" / "aggregated_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("network_size,pdr\n100,0.95\n", encoding="utf-8")

    paths = make_all_plots._collect_nested_csvs(results_dir, "aggregated_results.csv")

    assert paths == [csv_path]


def test_preflight_validate_plot_modules_lists_all_issues(monkeypatch, capsys) -> None:
    make_all_plots.set_log_level("debug")
    monkeypatch.setattr(
        make_all_plots,
        "_validate_plot_modules_use_save_figure",
        lambda: {
            "article_c.step1.plots.plot_S1": "ne passe pas par save_figure",
        },
    )
    monkeypatch.setattr(
        make_all_plots,
        "_validate_plot_modules_no_titles",
        lambda: {
            "article_c.step1.plots.plot_S2": "usage interdit de set_title/suptitle",
        },
    )

    invalid = make_all_plots._preflight_validate_plot_modules()

    captured = capsys.readouterr()
    assert invalid == {
        "article_c.step1.plots.plot_S1": "ne passe pas par save_figure",
        "article_c.step1.plots.plot_S2": "usage interdit de set_title/suptitle",
    }
    assert "modules de plots fautifs détectés avant exécution" in captured.out
    assert "article_c.step1.plots.plot_S1" in captured.out
    assert "article_c.step1.plots.plot_S2" in captured.out


def test_run_plot_module_requires_source_parameter(monkeypatch) -> None:
    class FakeModule:
        @staticmethod
        def main() -> None:
            return None

    monkeypatch.setattr(make_all_plots.importlib, "import_module", lambda _: FakeModule)

    try:
        make_all_plots._run_plot_module(
            "fake.module",
            source="by_size",
        )
    except TypeError as exc:
        assert "ignore la source contractuelle" in str(exc)
    else:
        raise AssertionError("Un module sans paramètre source doit échouer.")


def test_run_plot_module_logs_effective_source(monkeypatch) -> None:
    logged: list[str] = []

    class FakeModule:
        LAST_EFFECTIVE_SOURCE = "by_size"

        @staticmethod
        def main(source: str) -> None:
            assert source == "by_size"

    monkeypatch.setattr(make_all_plots.importlib, "import_module", lambda _: FakeModule)
    monkeypatch.setattr(make_all_plots, "log_info", logged.append)

    make_all_plots._run_plot_module("fake.module", source="by_size")

    assert logged == ["[fake.module] source effective=by_size"]


def test_run_plot_module_fails_if_effective_source_differs(monkeypatch) -> None:
    class FakeModule:
        LAST_EFFECTIVE_SOURCE = "aggregates"

        @staticmethod
        def main(source: str) -> None:
            assert source == "by_size"

    monkeypatch.setattr(make_all_plots.importlib, "import_module", lambda _: FakeModule)

    try:
        make_all_plots._run_plot_module("fake.module", source="by_size")
    except RuntimeError as exc:
        assert "source non contractuelle" in str(exc)
    else:
        raise AssertionError("Une source effective divergente doit échouer.")


def test_main_source_by_size_without_step1_aggregate(monkeypatch, tmp_path) -> None:
    step1_results = tmp_path / "step1" / "results"
    step1_csv = step1_results / "by_size" / "size_100" / "aggregated_results.csv"
    step1_csv.parent.mkdir(parents=True, exist_ok=True)
    step1_csv.write_text(
        "network_size,algo,snir_mode,cluster,pdr_mean\n100,adr,snir_on,all,0.9\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(make_all_plots, "STEP1_RESULTS_DIR", step1_results)
    monkeypatch.setattr(make_all_plots, "STEP1_PLOTS_OUTPUT_DIR", tmp_path / "step1" / "plots" / "output")
    monkeypatch.setattr(make_all_plots, "MANIFEST_OUTPUT_PATH", tmp_path / "figures_manifest.csv")
    monkeypatch.setattr(make_all_plots, "PLOT_DATA_FILTER_REPORT_OUTPUT_PATH", tmp_path / "plot_data_filter_report.csv")
    monkeypatch.setattr(make_all_plots, "LEGEND_CHECK_REPORT_OUTPUT_PATH", tmp_path / "legend_check_report.csv")
    monkeypatch.setattr(make_all_plots, "PLOT_MODULES", {"step1": ["fake.module"], "step2": []})
    monkeypatch.setattr(make_all_plots, "POST_PLOT_MODULES", [])
    monkeypatch.setattr(make_all_plots, "_preflight_validate_plot_modules", lambda: {})
    monkeypatch.setattr(make_all_plots, "_validate_step2_plot_module_registry", lambda: None)

    class FakeModule:
        LAST_EFFECTIVE_SOURCE = "by_size"

        @staticmethod
        def main(source: str, **_kwargs: object) -> None:
            assert source == "by_size"

    monkeypatch.setattr(make_all_plots.importlib, "import_module", lambda _: FakeModule)

    make_all_plots.main(["--steps", "step1", "--source", "by_size", "--skip-scientific-qa"])


def test_make_all_plots_post_modules_source_by_size(monkeypatch, tmp_path) -> None:
    for step in ("step1", "step2"):
        csv_path = (
            tmp_path
            / step
            / "results"
            / "by_size"
            / "size_100"
            / "aggregated_results.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "network_size,algo,snir_mode,cluster,pdr_mean,throughput_success_mean\n"
            "100,adr,snir_on,all,0.9,1.0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(make_all_plots, "STEP1_RESULTS_DIR", tmp_path / "step1" / "results")
    monkeypatch.setattr(make_all_plots, "STEP2_RESULTS_DIR", tmp_path / "step2" / "results")
    monkeypatch.setattr(make_all_plots, "STEP1_PLOTS_OUTPUT_DIR", tmp_path / "step1" / "plots" / "output")
    monkeypatch.setattr(make_all_plots, "STEP2_PLOTS_OUTPUT_DIR", tmp_path / "step2" / "plots" / "output")
    monkeypatch.setattr(make_all_plots, "MANIFEST_OUTPUT_PATH", tmp_path / "figures_manifest.csv")
    monkeypatch.setattr(make_all_plots, "PLOT_DATA_FILTER_REPORT_OUTPUT_PATH", tmp_path / "plot_data_filter_report.csv")
    monkeypatch.setattr(make_all_plots, "LEGEND_CHECK_REPORT_OUTPUT_PATH", tmp_path / "legend_check_report.csv")
    monkeypatch.setattr(make_all_plots, "PLOT_MODULES", {"step1": [], "step2": []})
    monkeypatch.setattr(
        make_all_plots,
        "POST_PLOT_MODULES",
        [
            "article_c.reproduce_author_results",
            "article_c.compare_with_snir",
            "article_c.plot_cluster_der",
        ],
    )
    monkeypatch.setattr(make_all_plots, "_preflight_validate_plot_modules", lambda: {})
    monkeypatch.setattr(make_all_plots, "_validate_step2_plot_module_registry", lambda: None)
    monkeypatch.setattr(make_all_plots, "_check_legends_for_module", lambda **_: [])

    seen_sources: dict[str, str] = {}


    def fake_import(module_path: str):
        module = type(module_path.split(".")[-1], (), {})
        module.LAST_EFFECTIVE_SOURCE = ""

        def _main(argv: list[str], source: str, **_kwargs: object) -> None:
            assert "--source" in argv
            assert argv[argv.index("--source") + 1] == "by_size"
            assert source == "by_size"
            module.LAST_EFFECTIVE_SOURCE = source
            seen_sources[module_path] = source

        module.main = staticmethod(_main)
        return module

    monkeypatch.setattr(make_all_plots.importlib, "import_module", fake_import)

    make_all_plots.main(["--source", "by_size", "--skip-scientific-qa"])

    assert seen_sources == {
        "article_c.reproduce_author_results": "by_size",
        "article_c.compare_with_snir": "by_size",
        "article_c.plot_cluster_der": "by_size",
    }
