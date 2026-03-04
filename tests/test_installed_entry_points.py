from __future__ import annotations

from importlib import metadata

import pytest


def test_mobilesfrdth_console_script_is_installed() -> None:
    try:
        distribution = metadata.distribution("LoRaFlexSim")
    except metadata.PackageNotFoundError:
        pytest.skip("Package LoRaFlexSim non installé dans cet environnement de test.")

    mobilesfrdth_entry_points = [
        ep for ep in distribution.entry_points if ep.group == "console_scripts" and ep.name == "mobilesfrdth"
    ]

    assert mobilesfrdth_entry_points, "Le point d'entrée 'mobilesfrdth' doit être installé."
    assert any(
        ep.value == "mobilesfrdth.cli:main" for ep in mobilesfrdth_entry_points
    ), "Le point d'entrée 'mobilesfrdth' doit pointer vers 'mobilesfrdth.cli:main'."
