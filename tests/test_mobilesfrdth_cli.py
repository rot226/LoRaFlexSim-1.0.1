import pathlib
import sys

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mobilesfrdth import cli


def test_main_rejects_unsupported_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 10, 9, "final", 0))

    exit_code = cli.main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Version Python non supportée" in captured.err
    assert "utiliser Python 3.11 ou 3.12" in captured.err


def test_main_help_supported_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
