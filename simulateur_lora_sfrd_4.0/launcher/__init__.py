"""Compatibility wrapper exposing the versioned launcher package.

This package simply re-exports everything from :mod:`VERSION_3.launcher` while
also making its submodules importable as ``launcher.<module>``.  Without this
extra step an ``ImportError`` would be raised when trying to import a module
such as ``launcher.simulator`` because the actual sources live under
``VERSION_3/launcher``.
"""

from __future__ import annotations

import os
from importlib import import_module

# Add the real package location to ``__path__`` so that ``import
# launcher.simulator`` works as expected.
_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "VERSION_3", "launcher")
__path__.append(os.path.abspath(_pkg_dir))

# Re-export public symbols from the versioned package.
module = import_module("VERSION_3.launcher")
__all__ = list(getattr(module, "__all__", []))
for name in __all__:
    globals()[name] = getattr(module, name)

