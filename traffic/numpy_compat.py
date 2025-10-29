"""Compatibility helpers between numpy and the lightweight test stub."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Tuple

import numpy as np


@lru_cache(None)
def generator_types() -> Tuple[type, ...]:
    """Return the accepted ``Generator`` classes from numpy or numpy_stub."""

    types: list[type] = []
    gen = getattr(np.random, "Generator", None)
    if gen is not None:
        types.append(gen)
    try:
        numpy_stub = importlib.import_module("numpy_stub")
    except ModuleNotFoundError:  # pragma: no cover - depends on test harness
        pass
    else:
        stub_gen = getattr(numpy_stub.random, "Generator", None)
        if stub_gen is not None:
            types.append(stub_gen)
    return tuple(types)


@lru_cache(None)
def bit_generator_types() -> Tuple[type, ...]:
    """Return the accepted bit generator classes (MT19937)."""

    types: list[type] = []
    mt = getattr(np.random, "MT19937", None)
    if mt is not None:
        types.append(mt)
    try:
        numpy_stub = importlib.import_module("numpy_stub")
    except ModuleNotFoundError:  # pragma: no cover - depends on test harness
        pass
    else:
        stub_mt = getattr(numpy_stub.random, "MT19937", None)
        if stub_mt is not None:
            types.append(stub_mt)
    return tuple(types)


def is_mt19937_rng(rng: object) -> bool:
    """Return ``True`` if ``rng`` mimics numpy's MT19937 generator."""

    bit_gen = getattr(rng, "bit_generator", None)
    if bit_gen is None:
        return False
    if type(bit_gen).__name__ != "MT19937":
        return False
    return callable(getattr(rng, "random", None))


def mt19937(seed: int | None = None):
    """Return an MT19937 bit generator from numpy or the stub."""

    try:
        numpy_stub = importlib.import_module("numpy_stub")
    except ModuleNotFoundError:  # pragma: no cover - depends on test harness
        return np.random.MT19937(seed)
    else:
        return numpy_stub.random.MT19937(seed)


def create_generator(seed: int | None = None):
    """Return a Generator backed by MT19937, preferring the stub when available."""

    bit_gen = mt19937(seed)
    try:
        numpy_stub = importlib.import_module("numpy_stub")
    except ModuleNotFoundError:  # pragma: no cover - depends on test harness
        return np.random.Generator(bit_gen)  # type: ignore[arg-type]
    else:
        return numpy_stub.random.Generator(bit_gen)


__all__ = [
    "generator_types",
    "bit_generator_types",
    "is_mt19937_rng",
    "mt19937",
    "create_generator",
]
