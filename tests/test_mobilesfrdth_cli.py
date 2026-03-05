import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mobilesfrdth import cli


def test_main_rejects_unsupported_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 10, 9, "final", 0))

    exit_code = cli.main(["run"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Version Python non supportée" in captured.err
    assert ">=3.11 et <3.13" in captured.err


def test_run_maps_to_sfrd_campaign_with_grid_preset(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))
    captured: dict[str, object] = {}

    def fake_campaign_main() -> None:
        captured["argv"] = sys.argv[:]

    monkeypatch.setattr(cli.run_campaign_module, "main", fake_campaign_main)

    exit_code = cli.main(["run", "--grid", "smoke"])

    stdio = capsys.readouterr()
    assert exit_code == 0
    assert "AVERTISSEMENT" in stdio.err
    assert "n'existe pas dans sfrd.cli.run_campaign" in stdio.err
    assert captured["argv"] == [
        "run_campaign",
        "--network-sizes",
        "80",
        "--algos",
        "ADR",
        "UCB",
        "--snir",
        "OFF,ON",
        "--replications",
        "1",
        "--seeds-base",
        "1",
        "--warmup-s",
        "300.0",
    ]


def test_run_rejects_grid_and_explicit_matrix(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))

    exit_code = cli.main(["run", "--grid", "smoke", "--network-sizes", "80"])

    stdio = capsys.readouterr()
    assert exit_code == 2
    assert "--grid est incompatible" in stdio.err


def test_aggregate_maps_to_sfrd_aggregate(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))
    captured: dict[str, object] = {}

    def fake_aggregate_main() -> None:
        captured["argv"] = sys.argv[:]

    monkeypatch.setattr(cli.aggregate, "main", fake_aggregate_main)

    exit_code = cli.main(["aggregate", "--logs-root", "tmp/logs", "--allow-partial"])

    assert exit_code == 0
    assert captured["argv"] == ["aggregate", "--logs-root", "tmp/logs", "--allow-partial"]


def test_plots_maps_to_sfrd_plot_campaign(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))
    captured: dict[str, object] = {}

    def fake_plot_main() -> None:
        captured["argv"] = sys.argv[:]

    monkeypatch.setattr(cli.plot_campaign, "main", fake_plot_main)

    exit_code = cli.main(["plots", "--campaign-id", "c01", "--format", "svg"])

    assert exit_code == 0
    assert captured["argv"] == ["plot_campaign", "--campaign-id", "c01", "--format", "svg"]
