"""Bandit UCB1 simple pour la sélection de SF."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ArmStats:
    """Statistiques empiriques par bras."""

    trials: int = 0
    mean_reward: float = 0.0


class UCBBandit:
    """Bandit UCB1 avec moyenne empirique et compteur d'essais."""

    def __init__(self, n_arms: int, *, exploration_coeff: float = 2.0) -> None:
        if n_arms <= 0:
            raise ValueError("Le nombre de bras doit être strictement positif.")
        self.n_arms = n_arms
        self.exploration_coeff = exploration_coeff
        self._stats: List[ArmStats] = [ArmStats() for _ in range(n_arms)]
        self.total_trials = 0

    def select_arm(self) -> int:
        """Retourne l'indice du bras à jouer selon l'indice UCB1."""

        for idx, stat in enumerate(self._stats):
            if stat.trials == 0:
                return idx

        log_total = math.log(self.total_trials)
        ucb_values = []
        for stat in self._stats:
            bonus = math.sqrt(self.exploration_coeff * log_total / stat.trials)
            ucb_values.append(stat.mean_reward + bonus)

        return int(max(range(self.n_arms), key=lambda arm: ucb_values[arm]))

    def update(self, arm: int, reward: float) -> None:
        """Met à jour la moyenne empirique du bras avec une récompense."""

        if arm < 0 or arm >= self.n_arms:
            raise IndexError("Indice de bras invalide.")
        self.total_trials += 1
        stat = self._stats[arm]
        stat.trials += 1
        stat.mean_reward += (reward - stat.mean_reward) / stat.trials

    @property
    def trials(self) -> List[int]:
        """Nombre d'essais par bras."""

        return [stat.trials for stat in self._stats]

    @property
    def mean_rewards(self) -> List[float]:
        """Moyenne empirique des récompenses par bras."""

        return [stat.mean_reward for stat in self._stats]
