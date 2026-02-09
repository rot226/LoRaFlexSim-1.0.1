"""Bandit UCB1 simple pour la sélection de SF."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ArmStats:
    """Statistiques empiriques par bras."""

    trials: float = 0.0
    mean_reward: float = 0.0


class UCBBandit:
    """Bandit UCB1 avec moyenne empirique et compteur d'essais."""

    def __init__(
        self,
        n_arms: int,
        *,
        exploration_coeff: float = 2.0,
        forgetting_factor: float | None = None,
        reset_on_change: bool = False,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("Le nombre de bras doit être strictement positif.")
        self.n_arms = n_arms
        self.exploration_coeff = exploration_coeff
        self._stats: List[ArmStats] = [ArmStats() for _ in range(n_arms)]
        self.total_trials = 0.0
        if forgetting_factor is not None and not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("Le facteur d'oubli doit être dans ]0, 1].")
        self.forgetting_factor = forgetting_factor
        self.reset_on_change = reset_on_change
        self._last_snir_state: bool | None = None
        self._last_cell_id: int | None = None

    def select_arm(self) -> int:
        """Retourne l'indice du bras à jouer selon l'indice UCB1."""

        for idx, stat in enumerate(self._stats):
            if stat.trials <= 0.0:
                return idx

        log_total = math.log(max(self.total_trials, 1.0))
        ucb_values = []
        for stat in self._stats:
            bonus = math.sqrt(self.exploration_coeff * log_total / max(stat.trials, 1e-9))
            ucb_values.append(stat.mean_reward + bonus)

        return int(max(range(self.n_arms), key=lambda arm: ucb_values[arm]))

    def update(self, arm: int, reward: float) -> None:
        """Met à jour la moyenne empirique du bras avec une récompense."""

        if arm < 0 or arm >= self.n_arms:
            raise IndexError("Indice de bras invalide.")
        self.total_trials += 1.0
        stat = self._stats[arm]
        stat.trials += 1.0
        stat.mean_reward += (reward - stat.mean_reward) / max(stat.trials, 1e-9)

    def notify_context_change(self, *, snir_state: bool | None = None, cell_id: int | None = None) -> bool:
        """Réagit à un changement de contexte (SNIR ou cellule)."""

        changed = False
        if snir_state is not None and snir_state != self._last_snir_state:
            changed = True
            self._last_snir_state = snir_state
        if cell_id is not None and cell_id != self._last_cell_id:
            changed = True
            self._last_cell_id = cell_id

        if changed:
            if self.reset_on_change:
                self.reset()
            elif self.forgetting_factor is not None:
                self.apply_forgetting(self.forgetting_factor)
        return changed

    def apply_forgetting(self, factor: float) -> None:
        """Applique un facteur d'oubli aux statistiques empiriques."""

        if not (0.0 < factor <= 1.0):
            raise ValueError("Le facteur d'oubli doit être dans ]0, 1].")
        self.total_trials *= factor
        for stat in self._stats:
            stat.trials *= factor

    def reset(self) -> None:
        """Réinitialise les statistiques du bandit."""

        self._stats = [ArmStats() for _ in range(self.n_arms)]
        self.total_trials = 0.0

    @property
    def trials(self) -> List[int]:
        """Nombre d'essais par bras."""

        return [int(round(stat.trials)) for stat in self._stats]

    @property
    def mean_rewards(self) -> List[float]:
        """Moyenne empirique des récompenses par bras."""

        return [stat.mean_reward for stat in self._stats]
