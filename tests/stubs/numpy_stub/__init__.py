"""Minimal NumPy stub used for tests when the real package is unavailable."""

from . import random

from datetime import datetime, timedelta

__all__ = [
    "random",
    "array",
    "zeros",
    "linspace",
    "diff",
    "histogram",
    "isscalar",
    "asarray",
    "ndarray",
    "datetime64",
    "timedelta64",
]


def array(obj, dtype=None):
    if hasattr(obj, "__iter__"):
        return list(obj)
    return [obj]


def zeros(shape, dtype=float):
    if isinstance(shape, int):
        return [0 for _ in range(shape)]
    if len(shape) == 2:
        rows, cols = shape
        return [[0 for _ in range(cols)] for _ in range(rows)]
    raise ValueError("unsupported shape")


def linspace(start, stop, num):
    if num <= 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def diff(a):
    return [a[i + 1] - a[i] for i in range(len(a) - 1)]


def histogram(a, bins=10):
    if not a:
        return [0] * bins, [0] * (bins + 1)
    lo, hi = min(a), max(a)
    if hi == lo:
        edges = [lo + i for i in range(bins + 1)]
        return [len(a)] + [0] * (bins - 1), edges
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    hist = [0] * bins
    for x in a:
        idx = int((x - lo) / width)
        if idx == bins:
            idx -= 1
        hist[idx] += 1
    return hist, edges


# Compatibility helpers for ``pytest`` which expects a few ``numpy`` APIs.
class ndarray(list):
    """Minimal stand‑in for :class:`numpy.ndarray`."""


def asarray(obj):
    """Return a list representation of *obj* as an ``ndarray``."""
    if isinstance(obj, ndarray):
        return obj
    return ndarray(array(obj))


def isscalar(obj) -> bool:
    """Return ``True`` if *obj* behaves like a scalar value."""
    return not isinstance(obj, (list, tuple, dict, set, ndarray))


# Alias used by ``pytest`` when checking for numpy booleans.
bool_ = bool


class _Datetime64(datetime):
    """Very small shim used by :mod:`pandas` during tests."""

    def __new__(cls, value):
        if isinstance(value, datetime):
            return datetime.__new__(cls, value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            return datetime.__new__(cls, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
        if isinstance(value, (int, float)):
            # Interpret numbers as seconds since epoch for the purposes of the tests.
            dt = datetime.fromtimestamp(value)
            return datetime.__new__(cls, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
        raise TypeError("Unsupported datetime64 input")


def datetime64(value, _unit=None):  # pragma: no cover - exercised indirectly
    return _Datetime64(value)


class _Timedelta64(timedelta):
    pass


def timedelta64(value, unit="s"):
    if isinstance(value, timedelta):
        return _Timedelta64(seconds=value.total_seconds())
    if unit == "s":
        return _Timedelta64(seconds=float(value))
    if unit == "ms":
        return _Timedelta64(milliseconds=float(value))
    if unit == "us":
        return _Timedelta64(microseconds=float(value))
    raise ValueError("Unsupported timedelta unit")
