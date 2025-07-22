import math
import random


class GaussMarkov:
    """Gauss–Markov mobility model with bounded area."""

    def __init__(
        self,
        area_size: float,
        mean_speed: float = 1.0,
        mean_direction: float | None = None,
        alpha: float = 0.75,
        speed_std: float = 0.2,
        direction_std: float = 0.2,
        step: float = 1.0,
    ) -> None:
        self.area_size = float(area_size)
        self.mean_speed = float(mean_speed)
        self.mean_direction = mean_direction if mean_direction is not None else random.random() * 2 * math.pi
        self.alpha = float(alpha)
        self.speed_std = float(speed_std)
        self.direction_std = float(direction_std)
        self.step = float(step)

    # ------------------------------------------------------------------
    def assign(self, node) -> None:
        node.speed = self.mean_speed
        node.direction = self.mean_direction
        node.vx = node.speed * math.cos(node.direction)
        node.vy = node.speed * math.sin(node.direction)
        node.last_move_time = 0.0

    # ------------------------------------------------------------------
    def _update_velocity(self, node) -> None:
        node.speed = (
            self.alpha * node.speed
            + (1 - self.alpha) * self.mean_speed
            + math.sqrt(1 - self.alpha ** 2) * random.gauss(0.0, self.speed_std)
        )
        node.direction = (
            self.alpha * node.direction
            + (1 - self.alpha) * self.mean_direction
            + math.sqrt(1 - self.alpha ** 2) * random.gauss(0.0, self.direction_std)
        )
        node.vx = node.speed * math.cos(node.direction)
        node.vy = node.speed * math.sin(node.direction)

    # ------------------------------------------------------------------
    def move(self, node, current_time: float) -> None:
        dt = current_time - node.last_move_time
        if dt <= 0:
            return
        steps = max(1, int(dt / self.step))
        remainder = dt - steps * self.step
        for _ in range(steps):
            self._update_velocity(node)
            node.x += node.vx * self.step
            node.y += node.vy * self.step
            if node.x < 0.0 or node.x > self.area_size:
                node.vx = -node.vx
                node.x = min(max(node.x, 0.0), self.area_size)
            if node.y < 0.0 or node.y > self.area_size:
                node.vy = -node.vy
                node.y = min(max(node.y, 0.0), self.area_size)
        if remainder > 0:
            self._update_velocity(node)
            node.x += node.vx * remainder
            node.y += node.vy * remainder
            if node.x < 0.0 or node.x > self.area_size:
                node.vx = -node.vx
                node.x = min(max(node.x, 0.0), self.area_size)
            if node.y < 0.0 or node.y > self.area_size:
                node.vy = -node.vy
                node.y = min(max(node.y, 0.0), self.area_size)
        node.last_move_time = current_time
