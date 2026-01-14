"""Implémentation minimale de l'algorithme UCB1."""


class BanditUCB1:
    def __init__(self, n_arms: int, warmup_rounds: int = 5) -> None:
        if n_arms <= 0:
            raise ValueError("Le nombre de bras doit être positif.")
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0
        self.warmup_rounds = warmup_rounds

    def select_arm(self) -> int:
        if self.t < self.warmup_rounds:
            return self.t % self.n_arms

        for idx, count in enumerate(self.counts):
            if count == 0:
                return idx

        import math

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
        self.counts[arm_index] += 1
        count = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = value + (reward - value) / count
