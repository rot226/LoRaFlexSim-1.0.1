"""Random Waypoint simplifié."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class RandomWaypointState:
    x: float
    y: float
    target_x: float
    target_y: float
    speed_mps: float


@dataclass(frozen=True)
class RandomWaypointConfig:
    area_width_m: float
    area_height_m: float
    min_speed_mps: float = 0.5
    max_speed_mps: float = 1.5


def _new_target(cfg: RandomWaypointConfig, rng: random.Random) -> tuple[float, float, float]:
    tx = rng.uniform(0.0, cfg.area_width_m)
    ty = rng.uniform(0.0, cfg.area_height_m)
    speed = rng.uniform(cfg.min_speed_mps, cfg.max_speed_mps)
    return tx, ty, speed


def init_state(cfg: RandomWaypointConfig, rng: random.Random | None = None) -> RandomWaypointState:
    generator = rng or random.Random()
    x = generator.uniform(0.0, cfg.area_width_m)
    y = generator.uniform(0.0, cfg.area_height_m)
    tx, ty, speed = _new_target(cfg, generator)
    return RandomWaypointState(x=x, y=y, target_x=tx, target_y=ty, speed_mps=speed)


def step(state: RandomWaypointState, dt_s: float, cfg: RandomWaypointConfig, rng: random.Random | None = None) -> None:
    generator = rng or random
    dx = state.target_x - state.x
    dy = state.target_y - state.y
    dist = math.hypot(dx, dy)
    travel = state.speed_mps * max(dt_s, 0.0)

    if dist <= 1e-9 or travel >= dist:
        state.x, state.y = state.target_x, state.target_y
        tx, ty, speed = _new_target(cfg, generator)
        state.target_x, state.target_y, state.speed_mps = tx, ty, speed
        return

    ratio = travel / dist
    state.x += dx * ratio
    state.y += dy * ratio
