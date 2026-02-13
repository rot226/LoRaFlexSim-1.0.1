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
