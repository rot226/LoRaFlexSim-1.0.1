from pathlib import Path
from typing import Iterable, List

from .mobility import RandomWaypoint as _BaseRandomWaypoint
from .map_loader import load_map


class RandomWaypoint(_BaseRandomWaypoint):
    """Random waypoint mobility loading terrain maps with :func:`load_map`."""

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
        step: float = 1.0,
        slope_scale: float = 0.1,
        dynamic_obstacles: List[dict[str, float]] | None = None,
    ) -> None:
        if terrain is not None and isinstance(terrain, (str, Path)):
            terrain = load_map(terrain)
        if elevation is not None and isinstance(elevation, (str, Path)):
            elevation = load_map(elevation)
        if obstacle_height_map is not None and isinstance(obstacle_height_map, (str, Path)):
            obstacle_height_map = load_map(obstacle_height_map)
        super().__init__(
            area_size,
            min_speed=min_speed,
            max_speed=max_speed,
            terrain=terrain,
            elevation=elevation,
            obstacle_height_map=obstacle_height_map,
            max_height=max_height,
            step=step,
            slope_scale=slope_scale,
            dynamic_obstacles=dynamic_obstacles,
        )
