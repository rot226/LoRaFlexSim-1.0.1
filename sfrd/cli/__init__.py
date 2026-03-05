"""Commandes CLI pour sfrd."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_legacy_cli_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "cli.py"
    spec = importlib.util.spec_from_file_location("sfrd._legacy_cli", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_campaign = _load_legacy_cli_module().run_campaign


def plot_campaign() -> Any:
    """Point d'entrée unique pour la génération de figures de campagne."""
    module = importlib.import_module("sfrd.cli.plot_campaign")
    return module.main()


__all__ = ["run_campaign", "plot_campaign"]
