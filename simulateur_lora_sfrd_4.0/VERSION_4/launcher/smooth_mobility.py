import math
import random


def bezier_point(p0, p1, p2, p3, t):
    """Return point of cubic Bezier curve for parameter t."""
    x = (
        ((1 - t) ** 3) * p0[0]
        + 3 * ((1 - t) ** 2) * t * p1[0]
        + 3 * (1 - t) * (t ** 2) * p2[0]
        + (t ** 3) * p3[0]
    )
    y = (
        ((1 - t) ** 3) * p0[1]
        + 3 * ((1 - t) ** 2) * t * p1[1]
        + 3 * (1 - t) * (t ** 2) * p2[1]
        + (t ** 3) * p3[1]
    )
    return x, y


class SmoothMobility:
    """Smooth node mobility based on cubic Bezier interpolation."""

    def __init__(self, area_size: float, min_speed: float = 2.0, max_speed: float = 10.0, step: float = 1.0):
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.step = step

    def assign(self, node):
        """Initialize path and speed for a node."""
        node.speed = float(random.uniform(self.min_speed, self.max_speed))
        node.path = self._generate_path(node.x, node.y)
        node.path_progress = 0.0
        node.path_duration = self._approx_length(node.path) / node.speed
        node.last_move_time = 0.0

    def _generate_path(self, x: float, y: float):
        start = (float(x), float(y))
        dest = (
            random.random() * self.area_size,
            random.random() * self.area_size,
        )
        offset = (
            (random.random() - 0.5) * (self.area_size * 0.1),
            (random.random() - 0.5) * (self.area_size * 0.1),
        )
        cp1 = (
            start[0] + (dest[0] - start[0]) / 3 + offset[0],
            start[1] + (dest[1] - start[1]) / 3 + offset[1],
        )
        cp2 = (
            start[0] + 2 * (dest[0] - start[0]) / 3 - offset[0],
            start[1] + 2 * (dest[1] - start[1]) / 3 - offset[1],
        )
        return start, cp1, cp2, dest

    def _approx_length(self, path, steps: int = 20) -> float:
        p0, p1, p2, p3 = path
        prev = bezier_point(p0, p1, p2, p3, 0.0)
        length = 0.0
        for i in range(1, steps + 1):
            t = i / steps
            pos = bezier_point(p0, p1, p2, p3, t)
            length += math.hypot(pos[0] - prev[0], pos[1] - prev[1])
            prev = pos
        return length

    def move(self, node, current_time: float):
        """Update node position according to the current Bezier path."""
        dt = current_time - node.last_move_time
        if dt <= 0:
            return
        node.path_progress += dt / node.path_duration
        while node.path_progress >= 1.0:
            # Reached the destination, start a new path
            node.x, node.y = map(float, node.path[3])
            node.path = self._generate_path(node.x, node.y)
            node.path_progress -= 1.0
            node.path_duration = self._approx_length(node.path) / node.speed
        t = node.path_progress
        p0, p1, p2, p3 = node.path
        pos = bezier_point(p0, p1, p2, p3, t)
        node.x, node.y = float(pos[0]), float(pos[1])
        node.last_move_time = current_time
