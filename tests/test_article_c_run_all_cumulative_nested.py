from __future__ import annotations

from article_c import run_all


def test_assert_cumulative_sizes_nested_detects_size_80_in_by_size_layout(tmp_path, capsys):
    results_dir = tmp_path / "results"
    for rep in range(1, 6):
        rep_dir = results_dir / "by_size" / "size_80" / f"rep_{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "aggregated_results.csv").write_text(
            "network_size\n80\n",
            encoding="utf-8",
        )

    run_all._assert_cumulative_sizes_nested(
        results_dir,
        expected_sizes_so_far={80},
        step_label="step-test",
    )

    stdout = capsys.readouterr().out
    assert "results/by_size/size_*" in stdout
    assert "size_80/rep_*" in stdout
