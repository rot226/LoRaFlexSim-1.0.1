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
    csv_path = results_dir / "by_size" / "size_100" / "rep_1" / "aggregated_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("network_size,pdr\n100,0.95\n", encoding="utf-8")

    paths = make_all_plots._collect_nested_csvs(results_dir, "aggregated_results.csv")

    assert paths == [csv_path]
