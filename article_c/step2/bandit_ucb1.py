"""Implémentation minimale de l'algorithme UCB1."""


class BanditUCB1:
    def __init__(
        self, n_arms: int, warmup_rounds: int = 5, epsilon_min: float = 0.05
    ) -> None:
        if n_arms <= 0:
            raise ValueError("Le nombre de bras doit être positif.")
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0
        self.non_zero_reward_rounds = 0
        self.warmup_rounds = warmup_rounds
        self.epsilon_min = max(0.0, min(1.0, epsilon_min))

    def select_arm(self) -> int:
        if self.t < self.warmup_rounds:
            return self.t % self.n_arms

        for idx, count in enumerate(self.counts):
            if count == 0:
                return idx

        import math
        import random

        if self.epsilon_min > 0.0 and random.random() < self.epsilon_min:
            return random.randrange(self.n_arms)

        total = max(1, self.t)
        scores = [
            self.values[idx] + math.sqrt(2 * math.log(total) / self.counts[idx])
            for idx in range(self.n_arms)
        ]
        return int(max(range(self.n_arms), key=scores.__getitem__))

    def update(self, arm_index: int, reward: float) -> None:
        if not 0 <= arm_index < self.n_arms:
            raise IndexError("Indice de bras invalide.")
        self.t += 1
        if reward > 0.0:
            self.non_zero_reward_rounds += 1
        self.counts[arm_index] += 1
        count = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = value + (reward - value) / count
