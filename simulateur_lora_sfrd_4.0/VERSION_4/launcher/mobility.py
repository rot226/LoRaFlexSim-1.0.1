import math
import random


class RandomWaypoint:
    """Modèle de mobilité aléatoire (Random Waypoint simplifié) pour les nœuds.

    Le modèle peut être couplé à un maillage représentant des obstacles ou un
    relief. Chaque cellule du maillage peut contenir un multiplicateur de
    vitesse (``1.0`` par défaut). Une valeur négative indique un obstacle
    infranchissable. Des obstacles dynamiques optionnels peuvent également être
    fournis. Les déplacements sont alors ralentis ou déviés en fonction de cette
    carte et de la topographie.
    """

    def __init__(
        self,
        area_size: float,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        *,
        terrain: list[list[float]] | None = None,
        elevation: list[list[float]] | None = None,
        obstacle_height_map: list[list[float]] | None = None,
        max_height: float = 0.0,
        step: float = 1.0,
        slope_scale: float = 0.1,
        dynamic_obstacles: list[dict[str, float]] | None = None,
    ) -> None:
        """
        Initialise le modèle de mobilité.
        :param area_size: Taille de l'aire carrée de simulation (mètres).
        :param min_speed: Vitesse minimale des nœuds (m/s).
        :param max_speed: Vitesse maximale des nœuds (m/s).
        """
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.step = step
        self.slope_scale = slope_scale
        self.terrain = terrain
        if terrain:
            self.rows = len(terrain)
            self.cols = len(terrain[0]) if self.rows else 0
        else:
            self.rows = 0
            self.cols = 0
        self.obstacle_height_map = obstacle_height_map
        self.max_height = max_height
        if obstacle_height_map:
            self.h_rows = len(obstacle_height_map)
            self.h_cols = len(obstacle_height_map[0]) if self.h_rows else 0
        else:
            self.h_rows = 0
            self.h_cols = 0
        self.elevation = elevation
        if elevation:
            self.e_rows = len(elevation)
            self.e_cols = len(elevation[0]) if self.e_rows else 0
        else:
            self.e_rows = 0
            self.e_cols = 0
        self.dynamic_obstacles = [dict(o) for o in (dynamic_obstacles or [])]
        self._last_obs_update = 0.0

    # ------------------------------------------------------------------
    def _terrain_factor(self, x: float, y: float) -> float | None:
        """Return the speed factor for coordinates or ``None`` if blocked."""
        if not self.terrain or self.rows == 0 or self.cols == 0:
            return 1.0
        cx = int(x / self.area_size * self.cols)
        cy = int(y / self.area_size * self.rows)
        cx = min(max(cx, 0), self.cols - 1)
        cy = min(max(cy, 0), self.rows - 1)
        val = float(self.terrain[cy][cx])
        if val < 0:
            return None
        return val if val > 0 else 1.0

    def _elevation(self, x: float, y: float) -> float:
        """Return the elevation value at coordinates (meters)."""
        if not self.elevation or self.e_rows == 0 or self.e_cols == 0:
            return 0.0
        cx = int(x / self.area_size * self.e_cols)
        cy = int(y / self.area_size * self.e_rows)
        cx = min(max(cx, 0), self.e_cols - 1)
        cy = min(max(cy, 0), self.e_rows - 1)
        return float(self.elevation[cy][cx])

    def _height(self, x: float, y: float) -> float:
        if not self.obstacle_height_map or self.h_rows == 0 or self.h_cols == 0:
            return 0.0
        cx = int(x / self.area_size * self.h_cols)
        cy = int(y / self.area_size * self.h_rows)
        cx = min(max(cx, 0), self.h_cols - 1)
        cy = min(max(cy, 0), self.h_rows - 1)
        return float(self.obstacle_height_map[cy][cx])

    def _update_dynamic_obstacles(self, dt: float) -> None:
        for obs in self.dynamic_obstacles:
            obs["x"] = float(obs.get("x", 0.0) + obs.get("vx", 0.0) * dt)
            obs["y"] = float(obs.get("y", 0.0) + obs.get("vy", 0.0) * dt)
            if obs["x"] < 0.0 or obs["x"] > self.area_size:
                obs["vx"] = -obs.get("vx", 0.0)
                obs["x"] = min(max(obs["x"], 0.0), self.area_size)
            if obs["y"] < 0.0 or obs["y"] > self.area_size:
                obs["vy"] = -obs.get("vy", 0.0)
                obs["y"] = min(max(obs["y"], 0.0), self.area_size)

    def _dynamic_blocked(self, x: float, y: float) -> bool:
        for obs in self.dynamic_obstacles:
            radius = float(obs.get("radius", 0.0))
            if math.hypot(x - obs.get("x", 0.0), y - obs.get("y", 0.0)) <= radius:
                return True
        return False

    def assign(self, node):
        """
        Assigne une direction et une vitesse aléatoires à un nœud.
        Initialise également son dernier temps de déplacement.
        """
        # Tirer un angle de direction uniforme dans [0, 2π) et une vitesse uniforme dans [min_speed, max_speed].
        angle = random.random() * 2 * math.pi
        speed = random.uniform(self.min_speed, self.max_speed)
        # Définir les composantes de vitesse selon la direction.
        node.vx = speed * math.cos(angle)
        node.vy = speed * math.sin(angle)
        node.speed = speed
        node.direction = angle
        # Initialiser le temps du dernier déplacement à 0 (début de simulation).
        node.last_move_time = 0.0

    def move(self, node, current_time: float):
        """
        Met à jour la position du nœud en le déplaçant selon sa vitesse et sa direction
        sur le laps de temps écoulé depuis son dernier déplacement, puis gère les rebonds aux bordures.
        :param node: Nœud à déplacer.
        :param current_time: Temps actuel de la simulation (secondes).
        """
        # Calculer le temps écoulé depuis le dernier déplacement
        dt = current_time - node.last_move_time
        if dt <= 0:
            return  # Pas de temps écoulé (ou appel redondant)
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles(current_time - self._last_obs_update)
            self._last_obs_update = current_time
        # Ajuster le déplacement selon la carte de vitesse
        factor = self._terrain_factor(node.x, node.y)
        if factor is None:
            factor = 1.0
        movement_factor = factor
        # Prendre en compte l'elevation si disponible
        if self.elevation:
            next_x = node.x + node.vx * dt * movement_factor
            next_y = node.y + node.vy * dt * movement_factor
            alt0 = self._elevation(node.x, node.y)
            alt1 = self._elevation(next_x, next_y)
            dist = math.hypot(next_x - node.x, next_y - node.y)
            if dist > 0:
                slope = (alt1 - alt0) / dist
                if slope > 0:
                    sf = 1.0 / (1.0 + slope * self.slope_scale)
                else:
                    sf = 1.0 + (-slope) * self.slope_scale * 0.5
                sf = max(0.1, min(2.0, sf))
                movement_factor *= sf
            node.altitude = alt0
        node.x += node.vx * dt * movement_factor
        node.y += node.vy * dt * movement_factor
        # Rebondir sur un obstacle infranchissable
        blocked = self._terrain_factor(node.x, node.y) is None
        if not blocked and self.dynamic_obstacles:
            blocked = self._dynamic_blocked(node.x, node.y)
        if not blocked and self.obstacle_height_map:
            if self._height(node.x, node.y) > self.max_height:
                blocked = True
        if blocked:
            node.x -= node.vx * dt * movement_factor
            node.y -= node.vy * dt * movement_factor
            node.vx = -node.vx
            node.vy = -node.vy
        # Gérer les rebonds sur les frontières de la zone [0, area_size]
        # Axe X
        if node.x < 0.0:
            node.x = -node.x  # symétrie par rapport au bord
            node.vx = -node.vx  # inversion de la direction X
        if node.x > self.area_size:
            node.x = 2 * self.area_size - node.x
            node.vx = -node.vx
        # Axe Y
        if node.y < 0.0:
            node.y = -node.y  # rebond sur le bord inférieur
            node.vy = -node.vy  # inversion de la direction Y
        if node.y > self.area_size:
            node.y = 2 * self.area_size - node.y
            node.vy = -node.vy
        # Mettre à jour la direction (angle) et la vitesse réelle
        node.direction = math.atan2(node.vy, node.vx)
        node.speed = math.hypot(node.vx, node.vy) * movement_factor
        # Mettre à jour l'altitude si une carte d'elevation est fournie
        if self.elevation:
            node.altitude = self._elevation(node.x, node.y)
        # Mettre à jour le temps du dernier déplacement du nœud
        node.last_move_time = current_time
