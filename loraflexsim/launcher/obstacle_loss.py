"""Simple obstacle attenuation model.

This module adds an additional path loss based on obstacles present
between a transmitter and a receiver. Obstacles can be loaded from a
GeoJSON file or from a raster matrix. The model is deliberately
light‑weight and avoids heavy geometric dependencies. Each obstacle is
represented by an axis‑aligned bounding box with an associated height and
material.

The attenuation for an intersected obstacle is computed as::

    loss = material_loss + 0.5 * height

where ``height`` is expressed in metres. Typical material losses are
provided in :data:`MATERIAL_LOSSES`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .map_loader import load_map


@dataclass
class _Obstacle:
    bbox: Tuple[float, float, float, float]
    height: float = 0.0
    material: str = "default"


class ObstacleLoss:
    """Compute additional loss due to obstacles.

    The class can be instantiated directly with a list of
    :class:`_Obstacle` or using the :meth:`from_geojson`,
    :meth:`from_raster` or :meth:`from_file` helpers.
    """

    MATERIAL_LOSSES = {
        "concrete": 15.0,
        "glass": 6.0,
        "wood": 5.0,
        "brick": 10.0,
        "steel": 20.0,
        "vegetation": 3.0,
        "default": 10.0,
    }

    def __init__(self, obstacles: Sequence[_Obstacle] | None = None) -> None:
        self.obstacles: List[_Obstacle] = list(obstacles or [])

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_from_coords(coords: Iterable) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []

        def _recurse(c: Iterable) -> None:
            if not c:
                return
            first = c[0]
            if isinstance(first, (list, tuple)):
                for sub in c:
                    _recurse(sub)
            else:
                x, y = c[:2]
                xs.append(float(x))
                ys.append(float(y))

        _recurse(coords)
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def from_geojson(cls, path: str | Path) -> "ObstacleLoss":
        """Create an :class:`ObstacleLoss` from a GeoJSON file.

        The geometry of each feature is approximated by its bounding box.
        Features may specify ``height`` and ``material`` properties.
        """
        data = json.loads(Path(path).read_text())
        obstacles: List[_Obstacle] = []
        for feat in data.get("features", []):
            geom = feat.get("geometry", {})
            bbox = cls._bbox_from_coords(geom.get("coordinates", []))
            props = feat.get("properties", {})
            height = float(props.get("height", 0.0))
            material = str(props.get("material", "default"))
            obstacles.append(_Obstacle(bbox, height, material))
        return cls(obstacles)

    @classmethod
    def from_raster(
        cls,
        raster: Iterable[Iterable[float]],
        *,
        cell_size: float = 1.0,
        material: str = "default",
    ) -> "ObstacleLoss":
        """Create from a raster matrix of heights.

        ``raster`` is a matrix where each cell represents the height of an
        obstacle in metres. ``cell_size`` defines the size of a cell in the
        same units as the coordinates used when computing the loss.
        Cells with a height ``<= 0`` are ignored.
        """
        obstacles: List[_Obstacle] = []
        for y, row in enumerate(raster):
            for x, val in enumerate(row):
                h = float(val)
                if h <= 0.0:
                    continue
                minx = x * cell_size
                miny = y * cell_size
                maxx = minx + cell_size
                maxy = miny + cell_size
                obstacles.append(_Obstacle((minx, miny, maxx, maxy), h, material))
        return cls(obstacles)

    @classmethod
    def from_file(cls, path: str | Path) -> "ObstacleLoss":
        """Load an obstacle map from a file.

        JSON/GeoJSON files are parsed using :meth:`from_geojson`. Any other
        extension is considered a plain text matrix and loaded with
        :func:`load_map`.
        """
        p = Path(path)
        if p.suffix.lower() in {".json", ".geojson"}:
            return cls.from_geojson(p)
        raster = load_map(p)
        return cls.from_raster(raster)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    @staticmethod
    def _segments_intersect(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> bool:
        def orient(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val > 0:
                return 1
            if val < 0:
                return 2
            return 0

        def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
            return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, q1, p2):
            return True
        if o2 == 0 and on_segment(p1, q2, p2):
            return True
        if o3 == 0 and on_segment(q1, p1, q2):
            return True
        if o4 == 0 and on_segment(q1, p2, q2):
            return True
        return False

    @classmethod
    def _line_intersects_bbox(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        minx, miny, maxx, maxy = bbox
        # Quick reject if both points on one side
        if (p1[0] < minx and p2[0] < minx) or (p1[0] > maxx and p2[0] > maxx):
            return False
        if (p1[1] < miny and p2[1] < miny) or (p1[1] > maxy and p2[1] > maxy):
            return False
        # Check if either point is inside
        if minx <= p1[0] <= maxx and miny <= p1[1] <= maxy:
            return True
        if minx <= p2[0] <= maxx and miny <= p2[1] <= maxy:
            return True
        # Check intersection with each edge of the rectangle
        edges = [
            ((minx, miny), (maxx, miny)),
            ((maxx, miny), (maxx, maxy)),
            ((maxx, maxy), (minx, maxy)),
            ((minx, maxy), (minx, miny)),
        ]
        return any(cls._segments_intersect(p1, p2, a, b) for a, b in edges)

    def loss(
        self,
        tx_pos: Tuple[float, float] | Sequence[float],
        rx_pos: Tuple[float, float] | Sequence[float],
    ) -> float:
        """Return additional loss between ``tx_pos`` and ``rx_pos``.

        Only the ``x`` and ``y`` coordinates of the provided positions are
        considered.
        """
        tx = (float(tx_pos[0]), float(tx_pos[1]))
        rx = (float(rx_pos[0]), float(rx_pos[1]))
        total = 0.0
        for obs in self.obstacles:
            if self._line_intersects_bbox(tx, rx, obs.bbox):
                base = self.MATERIAL_LOSSES.get(obs.material, self.MATERIAL_LOSSES["default"])
                total += base + 0.5 * obs.height
        return total


__all__ = ["ObstacleLoss"]
