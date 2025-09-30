from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .gps_mobility import GPSTraceMobility
from .map_loader import load_map


class Trace3DMobility(GPSTraceMobility):
    """Follow a time stamped trace while using 3D maps for altitude and obstacles."""

    def __init__(
        self,
        area_size: float,
        trace: str | Iterable[Sequence[float]],
        *,
        elevation: str | Path | Iterable[Iterable[float]] | None = None,
        obstacle_height_map: str | Path | Iterable[Iterable[float]] | None = None,
        max_height: float = 0.0,
        loop: bool = True,
    ) -> None:
        super().__init__(trace, loop=loop)
        self.area_size = float(area_size)
        if elevation is not None and not isinstance(elevation, list):
            elevation = load_map(elevation)
        self.elevation = elevation
        if elevation:
            self.e_rows = len(elevation)
            self.e_cols = len(elevation[0]) if self.e_rows else 0
        else:
            self.e_rows = self.e_cols = 0
        if obstacle_height_map is not None and not isinstance(obstacle_height_map, list):
            obstacle_height_map = load_map(obstacle_height_map)
        self.obstacle_height_map = obstacle_height_map
        self.max_height = float(max_height)
        if obstacle_height_map:
            self.h_rows = len(obstacle_height_map)
            self.h_cols = len(obstacle_height_map[0]) if self.h_rows else 0
        else:
            self.h_rows = self.h_cols = 0

    # ------------------------------------------------------------------
    def _elevation(self, x: float, y: float) -> float:
        if not self.elevation or self.e_rows == 0 or self.e_cols == 0:
            return 0.0
        cx = int(x / self.area_size * self.e_cols)
        cy = int(y / self.area_size * self.e_rows)
        cx = min(max(cx, 0), self.e_cols - 1)
        cy = min(max(cy, 0), self.e_rows - 1)
        return float(self.elevation[cy][cx])

    def _height(self, x: float, y: float) -> float:
        if not self.obstacle_height_map or self.h_rows == 0 or self.h_cols == 0:
            return 0.0
        cx = int(x / self.area_size * self.h_cols)
        cy = int(y / self.area_size * self.h_rows)
        cx = min(max(cx, 0), self.h_cols - 1)
        cy = min(max(cy, 0), self.h_rows - 1)
        return float(self.obstacle_height_map[cy][cx])

    # ------------------------------------------------------------------
    def move(self, node, current_time: float) -> None:
        prev_x, prev_y, prev_alt = node.x, node.y, node.altitude
        super().move(node, current_time)
        if self.elevation:
            node.altitude = self._elevation(node.x, node.y)
        if self.obstacle_height_map:
            height = self._height(node.x, node.y)
            if height > self.max_height and node.altitude <= height:
                node.x, node.y, node.altitude = prev_x, prev_y, prev_alt
            else:
                node.altitude = max(node.altitude, height)
