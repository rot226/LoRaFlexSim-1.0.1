"""Algorithme UCB1 pour la sélection de facteur d'étalement LoRa."""
from __future__ import annotations

import math
from typing import Dict, List


class UCB1Bandit:
    """Implémentation simple de l'algorithme UCB1.

    Les bras sont indexés de 0 à ``n_arms - 1``.
    """

    def __init__(self, n_arms: int = 6) -> None:
        self.n_arms = n_arms
        self.counts: List[int] = [0 for _ in range(self.n_arms)]
        self.values: List[float] = [0.0 for _ in range(self.n_arms)]
        self.total_rounds = 0

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

    def update(self, arm: int, reward: float) -> None:
        """Met à jour les statistiques du bras après avoir observé une récompense."""

        self.total_rounds += 1
        self.counts[arm] += 1
        count = self.counts[arm]
        value = self.values[arm]
        # Mise à jour incrémentale de la moyenne : new_value = old + (r - old) / n
        self.values[arm] = value + (reward - value) / count

    def reset(self) -> None:
        """Réinitialise l'état du bandit."""

        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
        self.total_rounds = 0


class LoRaSFSelectorUCB1:
    """Sélecteur de facteur d'étalement basé sur UCB1."""

    ARM_TO_SF: Dict[int, str] = {i: f"SF{7 + i}" for i in range(6)}
    SF_TO_ARM: Dict[str, int] = {sf: arm for arm, sf in ARM_TO_SF.items()}

    def __init__(self) -> None:
        self.bandit = UCB1Bandit(n_arms=6)

    def select_sf(self) -> str:
        """Retourne le facteur d'étalement à utiliser."""

        arm = self.bandit.select_arm()
        return self.ARM_TO_SF[arm]

    @staticmethod
    def reward_from_outcome(success: bool, snir_positive: bool | None = None) -> int:
        """Convertit le résultat radio en récompense binaire.

        Un succès radio vaut 1. Si ``success`` est ``False``, on peut considérer le SNIR
        comme un indice complémentaire : un SNIR positif peut être compté comme succès
        partiel. Par défaut, toute absence de succès produit une récompense nulle.
        """

        if success:
            return 1

        if snir_positive:
            return 1

        return 0

    def update(self, sf: str, reward: float) -> None:
        """Met à jour l'état du bandit à partir du facteur d'étalement choisi."""

        arm = self.SF_TO_ARM[sf]
        self.bandit.update(arm, reward)

    def reset(self) -> None:
        """Réinitialise les statistiques internes du sélecteur."""

        self.bandit.reset()
