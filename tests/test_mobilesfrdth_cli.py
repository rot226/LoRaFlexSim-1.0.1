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


def test_main_accepts_supported_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))

    exit_code = cli.main(["run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Sous-commande 'run' appelée." in captured.out
