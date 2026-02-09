"""Modèles de mobilité dédiés aux scénarios RL mobiles IWCMC."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from traffic.numpy_compat import create_generator


@dataclass(frozen=True)
class MobilitySpec:
    """Description d'un modèle de mobilité configurable."""

    key: str
    label: str


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class RandomWaypointMobility:
    """Random Waypoint simple (rebonds sur les frontières)."""

    def __init__(
        self,
        area_size: float,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        *,
        step: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.step = step
        self.rng = rng or create_generator()

    def assign(self, node) -> None:
        angle = float(self.rng.random() * 2 * math.pi)
        speed = float(self.min_speed + (self.max_speed - self.min_speed) * self.rng.random())
        node.vx = speed * math.cos(angle)
        node.vy = speed * math.sin(angle)
        node.speed = speed
        node.direction = angle
        node.last_move_time = 0.0

    def move(self, node, current_time: float) -> None:
        dt = current_time - node.last_move_time
        if dt <= 0:
            return
        node.x += node.vx * dt
        node.y += node.vy * dt

        if node.x < 0.0:
            node.x = -node.x
            node.vx = -node.vx
        if node.x > self.area_size:
            node.x = 2 * self.area_size - node.x
            node.vx = -node.vx
        if node.y < 0.0:
            node.y = -node.y
            node.vy = -node.vy
        if node.y > self.area_size:
            node.y = 2 * self.area_size - node.y
            node.vy = -node.vy

        node.direction = math.atan2(node.vy, node.vx)
        node.speed = math.hypot(node.vx, node.vy)
        node.last_move_time = current_time


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


class SmoothedKalmanMobility:
    """Mobilité lissée avec spline (Catmull-Rom) et filtre de Kalman."""

    def __init__(
        self,
        area_size: float,
        min_speed: float = 2.0,
        max_speed: float = 6.0,
        *,
        step: float = 1.0,
        kalman_q: float = 0.05,
        kalman_r: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.step = step
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.rng = rng or create_generator()

    def assign(self, node) -> None:
        node.speed = float(
            self.min_speed + (self.max_speed - self.min_speed) * self.rng.random()
        )
        node.path_points = self._init_points(node.x, node.y)
        node.path_progress = 0.0
        node.path_duration = self._segment_duration(node.path_points[1], node.path_points[2], node.speed)
        node.last_move_time = 0.0
        node.kalman_state = np.array([node.x, node.y, 0.0, 0.0], dtype=float)
        node.kalman_cov = np.eye(4, dtype=float)

    def _random_point(self) -> np.ndarray:
        return np.array(
            [self.rng.random() * self.area_size, self.rng.random() * self.area_size],
            dtype=float,
        )

    def _init_points(self, x: float, y: float) -> list[np.ndarray]:
        start = np.array([float(x), float(y)], dtype=float)
        p1 = self._random_point()
        p2 = self._random_point()
        p3 = self._random_point()
        return [start, p1, p2, p3]

    def _segment_duration(self, start: np.ndarray, end: np.ndarray, speed: float) -> float:
        distance = float(np.linalg.norm(end - start))
        return max(1.0, distance / max(speed, 1e-3))

    def _advance_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        return [points[1], points[2], points[3], self._random_point()]

    def _kalman_predict(self, state: np.ndarray, cov: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        q = np.eye(4, dtype=float) * float(self.kalman_q)
        next_state = f @ state
        next_cov = f @ cov @ f.T + q
        return next_state, next_cov

    def _kalman_update(
        self, state: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        r = np.eye(2, dtype=float) * float(self.kalman_r)
        y = measurement - (h @ state)
        s = h @ cov @ h.T + r
        k = cov @ h.T @ np.linalg.inv(s)
        updated_state = state + k @ y
        updated_cov = (np.eye(4, dtype=float) - k @ h) @ cov
        return updated_state, updated_cov

    def move(self, node, current_time: float) -> None:
        dt = current_time - node.last_move_time
        if dt <= 0:
            return

        node.path_progress += dt / node.path_duration
        while node.path_progress >= 1.0:
            node.path_progress -= 1.0
            node.path_points = self._advance_points(node.path_points)
            node.path_duration = self._segment_duration(
                node.path_points[1], node.path_points[2], node.speed
            )

        p0, p1, p2, p3 = node.path_points
        raw_pos = _catmull_rom(p0, p1, p2, p3, node.path_progress)

        state, cov = self._kalman_predict(node.kalman_state, node.kalman_cov, dt)
        state, cov = self._kalman_update(state, cov, raw_pos)
        node.kalman_state = state
        node.kalman_cov = cov

        node.x = _clamp(float(state[0]), 0.0, self.area_size)
        node.y = _clamp(float(state[1]), 0.0, self.area_size)
        node.last_move_time = current_time


def available_models() -> Iterable[Tuple[str, MobilitySpec]]:
    """Expose les modèles utilisables pour les scénarios mobiles."""

    specs = {
        "rwp": MobilitySpec(key="rwp", label="Random Waypoint"),
        "smooth": MobilitySpec(key="smooth", label="Spline + Kalman"),
    }
    return specs.items()
