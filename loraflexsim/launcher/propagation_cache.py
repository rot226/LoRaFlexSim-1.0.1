"""Caching helpers to reuse deterministic propagation computations."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Callable


class PropagationCache:
    """Thread-safe cache for distance dependent quantities.

    The cache quantises distances using ``resolution`` so that nearby values
    reuse the same entry.  This keeps the memory footprint bounded even in
    large scale simulations where the exact floating point distance may vary by
    a few millimetres due to numerical noise.
    """

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        max_entries: int | None = 10_000,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("resolution must be > 0")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be > 0 when provided")
        self.resolution = float(resolution)
        self.max_entries = max_entries
        self._lock = RLock()
        self._cache: "OrderedDict[int, float]" = OrderedDict()

    def _key(self, distance: float) -> int:
        if distance <= 0.0:
            raise ValueError("distance must be > 0")
        return int(round(distance / self.resolution))

    def get(self, distance: float, compute: Callable[[], float]) -> float:
        """Return a cached value for ``distance`` or compute it lazily."""

        key = self._key(distance)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = compute()
        with self._lock:
            self._cache[key] = value
            if self.max_entries is not None and len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop all cached values."""

        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._cache)


__all__ = ["PropagationCache"]

