"""Algorithme UCB1 pour la sélection de facteur d'étalement LoRa."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple


@dataclass(frozen=True)
class RewardSample:
    success: float
    snir: float
    energy: float
    collision: float
    fairness: float | None


class UCB1Bandit:
    """Implémentation simple de l'algorithme UCB1.

    Les bras sont indexés de 0 à ``n_arms - 1``.
    """

    def __init__(
        self, n_arms: int = 6, window_size: int = 20, *, traffic_weighted_mean: bool = False
    ) -> None:
        self.n_arms = n_arms
        self.counts: List[int] = [0 for _ in range(self.n_arms)]
        self.values: List[float] = [0.0 for _ in range(self.n_arms)]
        self.variances: List[float] = [0.0 for _ in range(self.n_arms)]
        self.total_rounds = 0
        self.window_size = max(1, window_size)
        self.traffic_weighted_mean = traffic_weighted_mean
        self._reward_windows: List[Deque[Tuple[float, float]]] = [
            deque(maxlen=self.window_size) for _ in range(self.n_arms)
        ]

    def select_arm(self) -> int:
        """Sélectionne le bras suivant en appliquant la borne supérieure de confiance.

        Retourne l'indice du bras à jouer.
        """

        # Exploration initiale : chaque bras doit être joué au moins une fois.
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm

        ucb_values = []
        for arm in range(self.n_arms):
            mean_reward = self.values[arm]
            confidence = math.sqrt(2 * math.log(self.total_rounds) / self.counts[arm])
            ucb_values.append(mean_reward + confidence)

        return int(max(range(self.n_arms), key=lambda arm: ucb_values[arm]))

    def update(self, arm: int, reward: float, *, weight: float = 1.0) -> None:
        """Met à jour les statistiques du bras après avoir observé une récompense."""

        self.total_rounds += 1
        self.counts[arm] += 1
        window = self._reward_windows[arm]
        window.append((reward, max(weight, 0.0)))
        window_mean = self._window_mean(window)
        window_variance = self._window_variance(window, window_mean)
        self.values[arm] = window_mean
        self.variances[arm] = window_variance

    def reset(self) -> None:
        """Réinitialise l'état du bandit."""

        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
        self.variances = [0.0 for _ in range(self.n_arms)]
        self.total_rounds = 0
        self._reward_windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    def _window_mean(self, window: Deque[Tuple[float, float]]) -> float:
        if not window:
            return 0.0
        if self.traffic_weighted_mean:
            total_weight = sum(weight for _, weight in window)
            if total_weight == 0:
                return 0.0
            return sum(reward * weight for reward, weight in window) / total_weight
        return sum(reward for reward, _ in window) / len(window)

    def _window_variance(self, window: Deque[Tuple[float, float]], mean: float) -> float:
        if not window:
            return 0.0
        if self.traffic_weighted_mean:
            total_weight = sum(weight for _, weight in window)
            if total_weight == 0:
                return 0.0
            return (
                sum(weight * (reward - mean) ** 2 for reward, weight in window)
                / total_weight
            )
        return sum((reward - mean) ** 2 for reward, _ in window) / len(window)

    @property
    def reward_window_mean(self) -> List[float]:
        """Retourne la moyenne des récompenses sur la fenêtre glissante."""

        return [self._window_mean(window) for window in self._reward_windows]

    @property
    def reward_window_variance(self) -> List[float]:
        """Retourne la variance des récompenses sur la fenêtre glissante."""

        return list(self.variances)


class LoRaSFSelectorUCB1:
    """Sélecteur de facteur d'étalement basé sur UCB1."""

    ARM_TO_SF: Dict[int, str] = {i: f"SF{7 + i}" for i in range(6)}
    SF_TO_ARM: Dict[str, int] = {sf: arm for arm, sf in ARM_TO_SF.items()}

    def __init__(
        self,
        *,
        success_weight: float = 1.0,
        snir_margin_weight: float = 0.1,
        fairness_weight: float = 0.0,
        energy_penalty_weight: float = 0.0,
        collision_penalty: float = 0.5,
        snir_threshold_db: float = 0.0,
        energy_normalization: float | None = None,
        reward_window: int = 20,
        traffic_weighted_mean: bool = False,
    ) -> None:
        self.bandit = UCB1Bandit(
            n_arms=6, window_size=reward_window, traffic_weighted_mean=traffic_weighted_mean
        )
        self.success_weight = success_weight
        self.snir_margin_weight = snir_margin_weight
        self.fairness_weight = fairness_weight
        self.energy_penalty_weight = energy_penalty_weight
        self.collision_penalty = collision_penalty
        self.snir_threshold_db = snir_threshold_db
        self.energy_normalization = energy_normalization
        self._qos_windows: List[Deque[RewardSample]] = [
            deque(maxlen=self.bandit.window_size) for _ in range(self.bandit.n_arms)
        ]

    def _compute_components(
        self,
        success: bool,
        *,
        snir_db: float | None = None,
        snir_threshold_db: float | None = None,
        marginal_snir_margin_db: float | None = None,
        airtime_s: float | None = None,
        energy_j: float | None = None,
        collision: bool | None = None,
        local_der: float | None = None,
        fairness_index: float | None = None,
        energy_normalization: float | None = None,
    ) -> RewardSample:
        success_component = local_der if local_der is not None else (1.0 if success else 0.0)
        success_component = min(max(success_component, 0.0), 1.0)

        threshold = self.snir_threshold_db if snir_threshold_db is None else snir_threshold_db
        snir_component = 0.0
        if snir_db is not None and threshold is not None:
            margin = snir_db - threshold
            snir_component = min(max(margin, 0.0), 1.0)
            if (
                marginal_snir_margin_db is not None
                and marginal_snir_margin_db > 0
                and 0.0 <= margin < marginal_snir_margin_db
            ):
                proximity = 1.0 - margin / marginal_snir_margin_db
                snir_component -= self.snir_margin_weight * proximity
            snir_component = max(min(snir_component, 1.0), -1.0)

        energy_metric = energy_j if energy_j is not None else airtime_s
        energy_component = 0.0
        normalization = energy_normalization
        if normalization is None:
            normalization = self.energy_normalization
        if energy_metric is not None:
            if normalization is not None and normalization > 0:
                energy_component = energy_metric / normalization
            else:
                energy_component = energy_metric
        energy_component = min(max(energy_component, 0.0), 1.0)

        collision_component = 1.0 if collision else 0.0

        fairness_component = None
        if fairness_index is not None:
            fairness_component = min(max(fairness_index, 0.0), 1.0)

        return RewardSample(
            success=success_component,
            snir=snir_component,
            energy=energy_component,
            collision=collision_component,
            fairness=fairness_component,
        )

    def _mean_components(self, window: Deque[RewardSample]) -> RewardSample:
        if not window:
            return RewardSample(success=0.0, snir=0.0, energy=0.0, collision=0.0, fairness=None)
        count = len(window)
        success = sum(sample.success for sample in window) / count
        snir = sum(sample.snir for sample in window) / count
        energy = sum(sample.energy for sample in window) / count
        collision = sum(sample.collision for sample in window) / count
        fairness_values = [sample.fairness for sample in window if sample.fairness is not None]
        fairness = None
        if fairness_values:
            fairness = sum(fairness_values) / len(fairness_values)
        return RewardSample(
            success=success,
            snir=snir,
            energy=energy,
            collision=collision,
            fairness=fairness,
        )

    def _combine_components(self, components: RewardSample, *, expected_der: float | None = None) -> float:
        expected = expected_der if expected_der is not None and expected_der > 0 else 1.0
        normalized_success = components.success
        if expected > 0:
            normalized_success = min(max(components.success / expected, 0.0), 1.0)

        reward = (
            self.success_weight * normalized_success
            + self.snir_margin_weight * components.snir
            - self.energy_penalty_weight * components.energy
            - self.collision_penalty * components.collision
        )
        if components.fairness is not None:
            reward += self.fairness_weight * components.fairness
        return reward

    def select_sf(self) -> str:
        """Retourne le facteur d'étalement à utiliser."""

        arm = self.bandit.select_arm()
        return self.ARM_TO_SF[arm]

    def reward_from_outcome(
        self,
        success: bool,
        *,
        snir_db: float | None = None,
        snir_threshold_db: float | None = None,
        marginal_snir_margin_db: float | None = None,
        airtime_s: float | None = None,
        energy_j: float | None = None,
        collision: bool | None = None,
        expected_der: float | None = None,
        local_der: float | None = None,
        fairness_index: float | None = None,
        energy_normalization: float | None = None,
    ) -> float:
        """Calcule une récompense combinant fiabilité, marge SNIR et coûts.

        Les pondérations sont configurables via le constructeur. Le paramètre
        ``snir_threshold_db`` permet d'utiliser un seuil spécifique, sinon la
        valeur par défaut fournie au constructeur est employée.
        """

        components = self._compute_components(
            success,
            snir_db=snir_db,
            snir_threshold_db=snir_threshold_db,
            marginal_snir_margin_db=marginal_snir_margin_db,
            airtime_s=airtime_s,
            energy_j=energy_j,
            collision=collision,
            local_der=local_der,
            fairness_index=fairness_index,
            energy_normalization=energy_normalization,
        )
        return self._combine_components(components, expected_der=expected_der)

    def update(
        self,
        sf: str,
        *,
        success: bool,
        snir_db: float | None = None,
        snir_threshold_db: float | None = None,
        airtime_s: float | None = None,
        energy_j: float | None = None,
        collision: bool | None = None,
        expected_der: float | None = None,
        local_der: float | None = None,
        traffic_volume: float | None = None,
        marginal_snir_margin_db: float | None = None,
        fairness_index: float | None = None,
        energy_normalization: float | None = None,
    ) -> float:
        """Met à jour l'état du bandit à partir du facteur d'étalement choisi."""

        arm = self.SF_TO_ARM[sf]
        sample = self._compute_components(
            success,
            snir_db=snir_db,
            snir_threshold_db=snir_threshold_db,
            marginal_snir_margin_db=marginal_snir_margin_db,
            airtime_s=airtime_s,
            energy_j=energy_j,
            collision=collision,
            local_der=local_der,
            fairness_index=fairness_index,
            energy_normalization=energy_normalization,
        )
        window = self._qos_windows[arm]
        window.append(sample)
        window_components = self._mean_components(window)
        window_reward = self._combine_components(window_components, expected_der=expected_der)

        weight = max(traffic_volume, 0.0) if traffic_volume is not None else 1.0
        self.bandit.update(arm, window_reward, weight=weight)
        return window_reward

    def reset(self) -> None:
        """Réinitialise les statistiques internes du sélecteur."""

        self.bandit.reset()
        self._qos_windows = [deque(maxlen=self.bandit.window_size) for _ in range(self.bandit.n_arms)]

    @property
    def reward_window_mean(self) -> List[float]:
        """Moyenne lissée des récompenses par bras."""

        return self.bandit.reward_window_mean

    @property
    def reward_window_variance(self) -> List[float]:
        """Variance lissée des récompenses par bras."""

        return self.bandit.reward_window_variance
