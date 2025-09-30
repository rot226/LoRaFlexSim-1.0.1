"""Custom mobility helpers with optional pause handling."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import RandomWaypoint as BaseRandomWaypoint
from loraflexsim.launcher import SmoothMobility as BaseSmoothMobility


class RandomWaypointWithPause(BaseRandomWaypoint):
    """Random waypoint mobility supporting configurable pauses."""

    def __init__(
        self,
        area_size: float,
        min_speed: float,
        max_speed: float,
        *,
        pause_mean: float = 0.0,
        step: float = 1.0,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            area_size,
            min_speed=min_speed,
            max_speed=max_speed,
            step=step,
            rng=rng,
            **kwargs,
        )
        self.pause_mean = max(0.0, float(pause_mean))
        self.step = step
        self.pause_rng = self.rng if rng is not None else np.random.Generator(np.random.MT19937())

    def assign(self, node: Any) -> None:
        super().assign(node)
        node.pause_remaining = 0.0

    def move(self, node: Any, current_time: float) -> None:
        last_time = getattr(node, "last_move_time", 0.0)
        dt = current_time - last_time
        if dt <= 0.0:
            return
        pause_remaining = float(getattr(node, "pause_remaining", 0.0))
        move_time = dt
        if pause_remaining > 0.0:
            if pause_remaining >= dt:
                node.pause_remaining = pause_remaining - dt
                node.last_move_time = current_time
                return
            move_time = dt - pause_remaining
            node.pause_remaining = 0.0
            last_time += pause_remaining
        target_time = last_time + move_time
        super().move(node, target_time)
        if self.pause_mean > 0.0:
            node.pause_remaining = float(self.pause_rng.exponential(self.pause_mean))
        else:
            node.pause_remaining = 0.0
        node.last_move_time = current_time


class SmoothMobilityWithPause(BaseSmoothMobility):
    """Smooth mobility model that inserts configurable pauses."""

    def __init__(
        self,
        area_size: float,
        min_speed: float,
        max_speed: float,
        *,
        pause_mean: float = 0.0,
        step: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            area_size,
            min_speed=min_speed,
            max_speed=max_speed,
            step=step,
            rng=rng,
        )
        self.pause_mean = max(0.0, float(pause_mean))
        self.pause_rng = self.rng if rng is not None else np.random.Generator(np.random.MT19937())

    def assign(self, node: Any) -> None:
        super().assign(node)
        node.pause_remaining = 0.0

    def move(self, node: Any, current_time: float) -> None:
        last_time = getattr(node, "last_move_time", 0.0)
        dt = current_time - last_time
        if dt <= 0.0:
            return
        pause_remaining = float(getattr(node, "pause_remaining", 0.0))
        move_time = dt
        if pause_remaining > 0.0:
            if pause_remaining >= dt:
                node.pause_remaining = pause_remaining - dt
                node.last_move_time = current_time
                return
            move_time = dt - pause_remaining
            node.pause_remaining = 0.0
            last_time += pause_remaining
        target_time = last_time + move_time
        super().move(node, target_time)
        if self.pause_mean > 0.0:
            node.pause_remaining = float(self.pause_rng.exponential(self.pause_mean))
        else:
            node.pause_remaining = 0.0
        node.last_move_time = current_time
