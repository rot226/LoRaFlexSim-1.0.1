import math
import random

class RandomWaypoint:
    """Modèle de mobilité aléatoire (Random Waypoint simplifié) pour les nœuds.

    Le modèle peut être couplé à un maillage représentant des obstacles ou un
    relief. Chaque cellule du maillage peut contenir un multiplicateur de
    vitesse (``1.0`` par défaut). Une valeur négative indique un obstacle
    infranchissable. Les déplacements sont alors ralentis ou déviés en fonction
    de cette carte.
    """

    def __init__(
        self,
        area_size: float,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        terrain: list[list[float]] | None = None,
        step: float = 1.0,
    ):
        """
        Initialise le modèle de mobilité.
        :param area_size: Taille de l'aire carrée de simulation (mètres).
        :param min_speed: Vitesse minimale des nœuds (m/s).
        :param max_speed: Vitesse maximale des nœuds (m/s).
        :param terrain: Carte influençant la vitesse ou bloquant les déplacements.
        :param step: Pas de temps entre deux mises à jour de position (s).
        """
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.terrain = terrain
        self.step = step
        if terrain:
            self.rows = len(terrain)
            self.cols = len(terrain[0]) if self.rows else 0
        else:
            self.rows = 0
            self.cols = 0

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
        # Ajuster le déplacement selon le relief/la carte
        factor = self._terrain_factor(node.x, node.y)
        if factor is None:
            factor = 1.0
        node.x += node.vx * dt * factor
        node.y += node.vy * dt * factor
        # Rebondir sur un obstacle infranchissable
        if self._terrain_factor(node.x, node.y) is None:
            node.x -= node.vx * dt * factor
            node.y -= node.vy * dt * factor
            node.vx = -node.vx
            node.vy = -node.vy
        # Gérer les rebonds sur les frontières de la zone [0, area_size]
        # Axe X
        if node.x < 0.0:
            node.x = -node.x               # symétrie par rapport au bord
            node.vx = -node.vx             # inversion de la direction X
        if node.x > self.area_size:
            node.x = 2 * self.area_size - node.x
            node.vx = -node.vx
        # Axe Y
        if node.y < 0.0:
            node.y = -node.y               # rebond sur le bord inférieur
            node.vy = -node.vy             # inversion de la direction Y
        if node.y > self.area_size:
            node.y = 2 * self.area_size - node.y
            node.vy = -node.vy
        # Mettre à jour la direction (angle) en cas de changement de vecteur vitesse
        node.direction = math.atan2(node.vy, node.vx)
        # Mettre à jour le temps du dernier déplacement du nœud
        node.last_move_time = current_time
