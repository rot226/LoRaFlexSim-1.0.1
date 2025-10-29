from pathlib import Path

from scripts import qos_cluster_plots


def test_generate_plots_without_results(tmp_path: Path):
    figures_dir = tmp_path / "figs"
    generated = qos_cluster_plots.generate_plots(tmp_path, figures_dir, quiet=True)
    assert generated is False
    assert not figures_dir.exists()
