from __future__ import annotations

import article_c.make_all_plots as make_all_plots


def test_validate_plot_modules_ignores_utils(monkeypatch) -> None:
    monkeypatch.setattr(
        make_all_plots,
        "PLOT_MODULES",
        {"step1": ["article_c.step1.plots.plot_S1"]},
    )

    assert make_all_plots._validate_plot_modules_use_save_figure() == {}
