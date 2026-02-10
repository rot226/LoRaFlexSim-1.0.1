"""Diagnose l'import du package article_c."""
from __future__ import annotations

import importlib
import os
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Optional


if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _resolve_module_path(module: ModuleType) -> Optional[str]:
    module_file = getattr(module, "__file__", None)
    if module_file:
        return os.path.abspath(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path:
        try:
            return os.path.abspath(next(iter(module_path)))
        except StopIteration:
            return None
    return None


def main() -> int:
    print("Diagnostic import 'article_c'")
    try:
        module = importlib.import_module("article_c")
    except Exception as exc:
        print("Échec de l'import :")
        print(f"  {exc.__class__.__name__}: {exc}")
        return 1

    resolved_path = _resolve_module_path(module)
    print("Import OK")
    print(f"Module: {module!r}")
    if resolved_path:
        print(f"Chemin résolu: {resolved_path}")
    else:
        print("Chemin résolu: <inconnu>")
    print("sys.path (extrait):")
    for entry in sys.path[:10]:
        print(f"  - {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
