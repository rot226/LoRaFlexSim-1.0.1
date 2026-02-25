"""Mobilité lissée (vitesse + direction avec inertie)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class SmoothState:
    x: float
    y: float
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True)
class SmoothConfig:
    area_width_m: float
    area_height_m: float
    min_speed_mps: float = 0.5
    max_speed_mps: float = 2.0
    heading_noise_std_rad: float = 0.1
    speed_relaxation: float = 0.15


def init_state(cfg: SmoothConfig, rng: random.Random | None = None) -> SmoothState:
    generator = rng or random.Random()
    return SmoothState(
        x=generator.uniform(0.0, cfg.area_width_m),
        y=generator.uniform(0.0, cfg.area_height_m),
        heading_rad=generator.uniform(-math.pi, math.pi),
        speed_mps=generator.uniform(cfg.min_speed_mps, cfg.max_speed_mps),
    )


def step(state: SmoothState, dt_s: float, cfg: SmoothConfig, rng: random.Random | None = None) -> None:
    generator = rng or random
    target_speed = generator.uniform(cfg.min_speed_mps, cfg.max_speed_mps)
    state.speed_mps += (target_speed - state.speed_mps) * max(0.0, min(cfg.speed_relaxation, 1.0))
    state.heading_rad += generator.gauss(0.0, cfg.heading_noise_std_rad)

    state.x += state.speed_mps * math.cos(state.heading_rad) * dt_s
    state.y += state.speed_mps * math.sin(state.heading_rad) * dt_s

    if state.x < 0.0 or state.x > cfg.area_width_m:
        state.heading_rad = math.pi - state.heading_rad
        state.x = min(max(state.x, 0.0), cfg.area_width_m)
    if state.y < 0.0 or state.y > cfg.area_height_m:
        state.heading_rad = -state.heading_rad
        state.y = min(max(state.y, 0.0), cfg.area_height_m)
