from __future__ import annotations

# Use a regular set instead of WeakSet because numpy.random.Generator
# does not support weak references in some numpy versions.
import hashlib
import importlib
import random
import secrets
from typing import Dict, Tuple

import numpy as np

from .numpy_compat import create_generator

try:
    _np_generator_module = importlib.import_module("numpy.random._generator")
except Exception:  # pragma: no cover - optional dependency
    _np_generator_module = None


class UncontrolledRandomError(RuntimeError):
    """Raised when an unmanaged RNG source is accessed."""


class RngManager:
    """Manage deterministic RNG streams based on MT19937."""

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        self._streams: Dict[Tuple[str, int], np.random.Generator] = {}

    def get_stream(self, stream_name: str, node_id: int = 0) -> np.random.Generator:
        """Return a Generator instance for the given stream and node."""
        key = (stream_name, node_id)
        if key not in self._streams:
            # ``hash()`` is not stable across interpreter runs so we
            # derive a deterministic hash from the stream name instead.
            digest = hashlib.sha256(stream_name.encode()).digest()
            stream_hash = int.from_bytes(digest[:8], "little")
            seed = (self.master_seed ^ stream_hash ^ node_id) & 0xFFFFFFFF
            gen = create_generator(seed)
            register_stream(gen)
            self._streams[key] = gen
        return self._streams[key]


_allowed_generators: "set[np.random.Generator]" = set()
_hook_enabled = False
_orig_random_funcs: dict[str, object] = {}
_orig_secret_funcs: dict[str, object] = {}
_orig_numpy_generator: type[np.random.Generator] | None = None


def register_stream(gen: np.random.Generator) -> None:
    """Mark ``gen`` as an allowed RNG stream."""

    _allowed_generators.add(gen)


def _reject(*_: object, **__: object) -> None:
    raise UncontrolledRandomError(
        "Unmanaged random source: use RngManager.get_stream()"
    )


def activate_global_hooks() -> None:
    """Globally reject uncontrolled RNG usage."""

    global _hook_enabled, _orig_numpy_generator
    if _hook_enabled:
        return
    _hook_enabled = True

    for name in [
        "random",
        "randrange",
        "randint",
        "choice",
        "shuffle",
        "uniform",
        "gauss",
        "betavariate",
        "expovariate",
        "gammavariate",
        "lognormvariate",
        "normalvariate",
        "paretovariate",
        "weibullvariate",
        "sample",
        "choices",
    ]:
        if hasattr(random, name):
            _orig_random_funcs[name] = getattr(random, name)
            setattr(random, name, _reject)

    _orig_numpy_generator = np.random.Generator

    class _HookedGenerator(_orig_numpy_generator):  # type: ignore[misc]
        def _check(self) -> None:
            if self not in _allowed_generators:
                _reject()

        def random(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().random(*a, **k)

        def normal(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().normal(*a, **k)

        def choice(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().choice(*a, **k)

        def integers(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().integers(*a, **k)

        def shuffle(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().shuffle(*a, **k)

        def standard_normal(self, *a, **k):  # type: ignore[override]
            self._check()
            return super().standard_normal(*a, **k)

    np.random.Generator = _HookedGenerator  # type: ignore[assignment]
    if _np_generator_module is not None:
        _np_generator_module.Generator = _HookedGenerator  # type: ignore[attr-defined]

    for name in [
        "choice",
        "randbelow",
        "randbits",
        "token_bytes",
        "token_hex",
        "token_urlsafe",
    ]:
        if hasattr(secrets, name):
            _orig_secret_funcs[name] = getattr(secrets, name)
            setattr(secrets, name, _reject)


def deactivate_global_hooks() -> None:
    """Restore modules to their original state."""

    global _hook_enabled, _orig_numpy_generator
    if not _hook_enabled:
        return

    for name, func in _orig_random_funcs.items():
        setattr(random, name, func)
    _orig_random_funcs.clear()

    for name, func in _orig_secret_funcs.items():
        setattr(secrets, name, func)
    _orig_secret_funcs.clear()

    if _orig_numpy_generator is not None:
        np.random.Generator = _orig_numpy_generator
        if _np_generator_module is not None:
            _np_generator_module.Generator = _orig_numpy_generator  # type: ignore[attr-defined]
        _orig_numpy_generator = None

    _hook_enabled = False
