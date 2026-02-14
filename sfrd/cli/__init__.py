"""Commandes CLI pour sfrd."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_legacy_cli_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "cli.py"
    spec = importlib.util.spec_from_file_location("sfrd._legacy_cli", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_campaign = _load_legacy_cli_module().run_campaign

__all__ = ["run_campaign"]
