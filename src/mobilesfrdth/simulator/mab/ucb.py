"""UCB1 stationnaire."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class UCB1:
    n_arms: int
    counts: list[int] = field(init=False)
    values: list[float] = field(init=False)
    total_pulls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]

    def select_arm(self) -> int:
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm

        log_t = math.log(max(self.total_pulls, 1))
        ucb_values = [
            self.values[arm] + math.sqrt(2.0 * log_t / self.counts[arm])
            for arm in range(self.n_arms)
        ]
        return max(range(self.n_arms), key=lambda a: ucb_values[a])

    def update(self, arm: int, reward: float) -> None:
        self.total_pulls += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
