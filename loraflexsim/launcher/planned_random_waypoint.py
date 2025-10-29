from pathlib import Path
from typing import Iterable
import math
import numpy as np

from traffic.numpy_compat import create_generator

from .waypoint_planner import WaypointPlanner3D
from .map_loader import load_map


class PlannedRandomWaypoint:
    """Random waypoint mobility using a 3D path planner."""

    def __init__(
        self,
        area_size: float,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        *,
        terrain: str | Path | Iterable[Iterable[float]] | None = None,
        elevation: str | Path | Iterable[Iterable[float]] | None = None,
        obstacle_height_map: str | Path | Iterable[Iterable[float]] | None = None,
        max_height: float = 0.0,
        slope_scale: float = 0.1,
        slope_limit: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise le modèle de mobilité planifié."""
        if terrain is None:
            raise ValueError("terrain map is required for planned random waypoint")
        if isinstance(terrain, (str, Path)):
            terrain = load_map(terrain)
        if elevation is not None and isinstance(elevation, (str, Path)):
            elevation = load_map(elevation)
        if obstacle_height_map is not None and isinstance(obstacle_height_map, (str, Path)):
            obstacle_height_map = load_map(obstacle_height_map)
        self.rng = rng or create_generator()
        self.planner = WaypointPlanner3D(
            area_size,
            terrain,
            elevation=elevation,
            obstacle_height_map=obstacle_height_map,
            max_height=max_height,
            slope_scale=slope_scale,
            slope_limit=slope_limit,
            rng=self.rng,
        )
        self.area_size = float(area_size)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)

    # --------------------------------------------------------------
    def assign(self, node) -> None:
        node.speed = float(
            self.min_speed + (self.max_speed - self.min_speed) * self.rng.random()
        )
        goal = self.planner.random_free_point()
        node.path = self.planner.find_path((node.x, node.y), goal)
        node.path_index = 0
        node.last_move_time = 0.0
        node.altitude = self.planner.elevation_at(node.x, node.y)

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
            if seg_len == 0:
                node.path_index += 1
                continue
            move = min(distance, seg_len)
            ratio = move / seg_len
            node.x += dx * ratio
            node.y += dy * ratio
            distance -= move
            if ratio >= 1.0:
                node.path_index += 1
        if node.path_index >= len(node.path) - 1:
            goal = self.planner.random_free_point()
            node.path = self.planner.find_path((node.x, node.y), goal)
            node.path_index = 0
        node.altitude = self.planner.elevation_at(node.x, node.y)
        node.last_move_time = current_time
