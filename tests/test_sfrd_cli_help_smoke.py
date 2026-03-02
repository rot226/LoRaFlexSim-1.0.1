from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "sfrd.cli.run_campaign",
    "sfrd.cli.validate_outputs",
    "sfrd.cli.calibrate_ucb",
    "sfrd.cli.check_trends",
    "sfrd.parse.aggregate",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_sfrd_modules_help_smoke(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage" in result.stdout.lower()
