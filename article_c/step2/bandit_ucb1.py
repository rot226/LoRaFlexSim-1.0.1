"""Implémentation minimale de l'algorithme UCB1 (placeholder)."""

from dataclasses import dataclass


@dataclass
class Arm:
    pulls: int = 0
    reward_sum: float = 0.0

    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


class UCB1:
    def __init__(self, n_arms: int) -> None:
        self.arms = [Arm() for _ in range(n_arms)]
        self.total_pulls = 0

    def select_arm(self) -> int:
        for idx, arm in enumerate(self.arms):
            if arm.pulls == 0:
                return idx
        import math

        scores = [
            arm.mean_reward() + math.sqrt(2 * math.log(self.total_pulls) / arm.pulls)
            for arm in self.arms
        ]
        return int(max(range(len(scores)), key=scores.__getitem__))

    def update(self, arm_index: int, reward: float) -> None:
        arm = self.arms[arm_index]
        arm.pulls += 1
        arm.reward_sum += reward
        self.total_pulls += 1
