# Path planner using A* over a terrain map with optional elevation and 3D obstacles.
import math
import numpy as np
from typing import List, Tuple


class WaypointPlanner3D:
    """A* planner avoiding obstacles and high buildings."""

    def __init__(
        self,
        area_size: float,
        terrain: List[List[float]],
        *,
        elevation: List[List[float]] | None = None,
        obstacle_height_map: List[List[float]] | None = None,
        max_height: float = 0.0,
        slope_scale: float = 0.1,
        slope_limit: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.area_size = float(area_size)
        self.terrain = terrain
        self.rows = len(terrain)
        self.cols = len(terrain[0]) if self.rows else 0
        self.elevation = elevation
        if elevation:
            self.e_rows = len(elevation)
            self.e_cols = len(elevation[0]) if self.e_rows else 0
        else:
            self.e_rows = self.e_cols = 0
        self.obstacle_height_map = obstacle_height_map
        self.max_height = max_height
        if obstacle_height_map:
            self.h_rows = len(obstacle_height_map)
            self.h_cols = len(obstacle_height_map[0]) if self.h_rows else 0
        else:
            self.h_rows = self.h_cols = 0
        self.slope_scale = slope_scale
        self.slope_limit = slope_limit
        self.rng = rng or np.random.Generator(np.random.MT19937())

    # --------------------------------------------------------------
    def _terrain_factor_cell(self, cx: int, cy: int) -> float | None:
        val = float(self.terrain[cy][cx])
        if val < 0:
            return None
        return val if val > 0 else 1.0

    def _height_cell(self, cx: int, cy: int) -> float:
        if not self.obstacle_height_map or self.h_rows == 0 or self.h_cols == 0:
            return 0.0
        hx = int(cx / self.cols * self.h_cols)
        hy = int(cy / self.rows * self.h_rows)
        hx = min(max(hx, 0), self.h_cols - 1)
        hy = min(max(hy, 0), self.h_rows - 1)
        return float(self.obstacle_height_map[hy][hx])

    def _elevation_cell(self, cx: int, cy: int) -> float:
        if not self.elevation or self.e_rows == 0 or self.e_cols == 0:
            return 0.0
        ex = int(cx / self.cols * self.e_cols)
        ey = int(cy / self.rows * self.e_rows)
        ex = min(max(ex, 0), self.e_cols - 1)
        ey = min(max(ey, 0), self.e_rows - 1)
        return float(self.elevation[ey][ex])

    def _coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(x / self.area_size * self.cols)
        cy = int(y / self.area_size * self.rows)
        cx = min(max(cx, 0), self.cols - 1)
        cy = min(max(cy, 0), self.rows - 1)
        return cx, cy

    def _cell_to_coord(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        x = (cell[0] + 0.5) / self.cols * self.area_size
        y = (cell[1] + 0.5) / self.rows * self.area_size
        return x, y

    # --------------------------------------------------------------
    def _neighbors(self, cell: Tuple[int, int]):
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if self._terrain_factor_cell(nx, ny) is not None:
                    if self._height_cell(nx, ny) <= self.max_height:
                        if self.slope_limit is not None and self.elevation:
                            alt_a = self._elevation_cell(x, y)
                            alt_b = self._elevation_cell(nx, ny)
                            dist = math.hypot(dx, dy)
                            if dist > 0:
                                slope = (alt_b - alt_a) / dist
                                if abs(slope) > self.slope_limit:
                                    continue
                        yield nx, ny

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        factor = self._terrain_factor_cell(b[0], b[1])
        if factor is None:
            return float("inf")
        cost = 1.0 / factor
        if self.elevation:
            alt_a = self._elevation_cell(*a)
            alt_b = self._elevation_cell(*b)
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            if dist > 0:
                slope = (alt_b - alt_a) / dist
                if self.slope_limit is not None and abs(slope) > self.slope_limit:
                    return float("inf")
                if slope > 0:
                    cost *= 1.0 + slope * self.slope_scale
        return cost

    # --------------------------------------------------------------
    def find_path(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        s = self._coord_to_cell(*start)
        g = self._coord_to_cell(*goal)
        open_set = {s}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {s: 0.0}
        f_score = {s: self._heuristic(s, g)}
        while open_set:
            current = min(open_set, key=lambda c: f_score.get(c, float("inf")))
            if current == g:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return [self._cell_to_coord(c) for c in cells]
            open_set.remove(current)
            for nb in self._neighbors(current):
                tentative_g = g_score[current] + self._cost(current, nb)
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score[nb] = tentative_g + self._heuristic(nb, g)
                    open_set.add(nb)
        return [start, goal]

    def random_free_point(self) -> Tuple[float, float]:
        while True:
            cx = int(self.rng.integers(self.cols))
            cy = int(self.rng.integers(self.rows))
            if self._terrain_factor_cell(cx, cy) is not None:
                if self._height_cell(cx, cy) <= self.max_height:
                    return self._cell_to_coord((cx, cy))

    def elevation_at(self, x: float, y: float) -> float:
        if not self.elevation:
            return 0.0
        cx = int(x / self.area_size * self.e_cols)
        cy = int(y / self.area_size * self.e_rows)
        cx = min(max(cx, 0), self.e_cols - 1)
        cy = min(max(cy, 0), self.e_rows - 1)
        return float(self.elevation[cy][cx])
