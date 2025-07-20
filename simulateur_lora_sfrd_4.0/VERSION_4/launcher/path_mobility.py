import heapq
import math
import random
from typing import Iterable, Tuple, List


class PathMapMobility:
    """Grid based mobility using A* path finding."""

    def __init__(
        self,
        map_data: Iterable[Iterable[float]],
        area_size: float,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
    ) -> None:
        self.map: List[List[float]] = [list(row) for row in map_data]
        self.area_size = float(area_size)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.rows = len(self.map)
        self.cols = len(self.map[0]) if self.rows else 0
        self.cell_w = self.area_size / self.cols if self.cols else 1.0
        self.cell_h = self.area_size / self.rows if self.rows else 1.0

    # --------------------------------------------------------------
    def _cell_from_pos(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(x / self.cell_w)
        cy = int(y / self.cell_h)
        cx = max(0, min(self.cols - 1, cx))
        cy = max(0, min(self.rows - 1, cy))
        return cx, cy

    def _pos_from_cell(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        cx, cy = cell
        return (cx + 0.5) * self.cell_w, (cy + 0.5) * self.cell_h

    # --------------------------------------------------------------
    def _neighbors(self, cell: Tuple[int, int]):
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                yield nx, ny

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self, start: Tuple[int, int], goal: Tuple[int, int]):
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g: dict[Tuple[int, int], float] = {start: 0.0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for nb in self._neighbors(current):
                if self.map[nb[1]][nb[0]] <= 0:
                    continue
                tentative = g[current] + 1
                if tentative < g.get(nb, float("inf")):
                    came_from[nb] = current
                    g[nb] = tentative
                    f = tentative + self._heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))
        return [start]

    # --------------------------------------------------------------
    def compute_path(
        self,
        x: float,
        y: float,
        dest: Tuple[float, float] | None = None,
    ) -> List[Tuple[float, float]]:
        if dest is None:
            while True:
                dest = (
                    random.random() * self.area_size,
                    random.random() * self.area_size,
                )
                cx, cy = self._cell_from_pos(dest[0], dest[1])
                if self.map[cy][cx] > 0:
                    break
        start_cell = self._cell_from_pos(x, y)
        goal_cell = self._cell_from_pos(dest[0], dest[1])
        cells = self._a_star(start_cell, goal_cell)
        return [self._pos_from_cell(c) for c in cells[1:]]

    # --------------------------------------------------------------
    def assign(self, node) -> None:
        node.speed = float(random.uniform(self.min_speed, self.max_speed))
        node.path = self.compute_path(node.x, node.y)
        node.last_move_time = 0.0

    # --------------------------------------------------------------
    def move(self, node, current_time: float) -> None:
        dt = current_time - node.last_move_time
        if dt <= 0:
            return
        remain = node.speed * dt
        while remain > 0:
            if not node.path:
                node.path = self.compute_path(node.x, node.y)
            next_x, next_y = node.path[0]
            dist = math.hypot(next_x - node.x, next_y - node.y)
            if dist <= remain:
                node.x, node.y = float(next_x), float(next_y)
                node.path.pop(0)
                remain -= dist
            else:
                ratio = remain / dist
                node.x += (next_x - node.x) * ratio
                node.y += (next_y - node.y) * ratio
                remain = 0.0
        node.last_move_time = current_time

