import math
import random

from typing import List, Tuple


class PathMobility:
    """Simple grid-based mobility that plans shortest paths avoiding obstacles."""

    def __init__(
        self,
        area_size: float,
        path_map: List[List[float]],
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        *,
        elevation: List[List[float]] | None = None,
        obstacle_height_map: List[List[float]] | None = None,
        max_height: float = 0.0,
        step: float = 1.0,
    ) -> None:
        self.area_size = float(area_size)
        self.path_map = path_map
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.step = float(step)
        self.rows = len(path_map)
        self.cols = len(path_map[0]) if self.rows else 0
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

    # ------------------------------------------------------------------
    def _height_cell(self, cx: int, cy: int) -> float:
        if not self.obstacle_height_map or self.h_rows == 0 or self.h_cols == 0:
            return 0.0
        hx = int(cx / self.cols * self.h_cols)
        hy = int(cy / self.rows * self.h_rows)
        hx = min(max(hx, 0), self.h_cols - 1)
        hy = min(max(hy, 0), self.h_rows - 1)
        return float(self.obstacle_height_map[hy][hx])

    def _elevation(self, x: float, y: float) -> float:
        if not self.elevation or self.e_rows == 0 or self.e_cols == 0:
            return 0.0
        cx = int(x / self.area_size * self.e_cols)
        cy = int(y / self.area_size * self.e_rows)
        cx = min(max(cx, 0), self.e_cols - 1)
        cy = min(max(cy, 0), self.e_rows - 1)
        return float(self.elevation[cy][cx])

    # ------------------------------------------------------------------
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

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, cell: Tuple[int, int]):
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if self.path_map[ny][nx] >= 0:
                    if self._height_cell(nx, ny) <= self.max_height:
                        yield nx, ny

    def _find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        open_set = {start}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}
        while open_set:
            current = min(open_set, key=lambda c: f_score.get(c, float("inf")))
            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return [self._cell_to_coord(c) for c in cells]
            open_set.remove(current)
            for nb in self._neighbors(current):
                tentative_g = g_score[current] + 1.0
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score[nb] = tentative_g + self._heuristic(nb, goal)
                    open_set.add(nb)
        return []

    def _random_free_cell(self) -> Tuple[int, int]:
        while True:
            cx = random.randrange(self.cols)
            cy = random.randrange(self.rows)
            if self.path_map[cy][cx] >= 0 and self._height_cell(cx, cy) <= self.max_height:
                return cx, cy

    def _new_path(self, x: float, y: float) -> List[Tuple[float, float]]:
        start = self._coord_to_cell(x, y)
        for _ in range(20):
            goal = self._random_free_cell()
            path = self._find_path(start, goal)
            if path:
                return path
        return [self._cell_to_coord(start)]

    # ------------------------------------------------------------------
    def assign(self, node):
        node.speed = float(random.uniform(self.min_speed, self.max_speed))
        node.path = self._new_path(node.x, node.y)
        node.path_index = 0
        node.last_move_time = 0.0
        node.altitude = self._elevation(node.x, node.y)

    def move(self, node, current_time: float) -> None:
        dt = current_time - node.last_move_time
        if dt <= 0:
            return
        distance = dt * node.speed
        while distance > 0 and node.path_index < len(node.path) - 1:
            dest_x, dest_y = node.path[node.path_index + 1]
            dx = dest_x - node.x
            dy = dest_y - node.y
            seg_len = math.hypot(dx, dy)
            if distance >= seg_len:
                node.x = dest_x
                node.y = dest_y
                node.path_index += 1
                distance -= seg_len
            else:
                ratio = distance / seg_len
                node.x += dx * ratio
                node.y += dy * ratio
                distance = 0
        if node.path_index >= len(node.path) - 1:
            node.path = self._new_path(node.x, node.y)
            node.path_index = 0
        node.altitude = self._elevation(node.x, node.y)
        node.last_move_time = current_time
