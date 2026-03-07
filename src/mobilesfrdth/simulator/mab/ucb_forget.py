"""UCB non stationnaire: discounted ou sliding-window."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class UCBForget:
    n_arms: int
    mode: str = "discounted"  # discounted | sliding_window
    gamma: float = 0.95
    window_size: int = 100
    discounted_counts: list[float] = field(init=False)
    discounted_rewards: list[float] = field(init=False)
    windows: list[deque[float]] = field(init=False)
    total_pulls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"discounted", "sliding_window"}:
            raise ValueError("mode doit être 'discounted' ou 'sliding_window'.")
        self.discounted_counts = [0.0 for _ in range(self.n_arms)]
        self.discounted_rewards = [0.0 for _ in range(self.n_arms)]
        self.windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    @classmethod
    def from_yaml_config(cls, cfg: dict) -> "UCBForget":
        """Construit l'agent depuis un dictionnaire YAML déjà parsé."""

        mab_cfg = cfg.get("mab", cfg)
        return cls(
            n_arms=int(mab_cfg["n_arms"]),
            mode=str(mab_cfg.get("mode", "discounted")),
            gamma=float(mab_cfg.get("gamma", 0.95)),
            window_size=int(mab_cfg.get("window_size", 100)),
        )

    def _arm_stats(self, arm: int) -> tuple[float, float]:
        if self.mode == "discounted":
            count = max(self.discounted_counts[arm], 1e-9)
            mean = self.discounted_rewards[arm] / count
            return count, mean
        count = len(self.windows[arm])
        if count == 0:
            return 0.0, 0.0
        return float(count), sum(self.windows[arm]) / count

    def select_arm(self) -> int:
        for arm in range(self.n_arms):
            count, _ = self._arm_stats(arm)
            if count <= 0:
                return arm

        log_t = math.log(max(self.total_pulls, 1))
        scores = []
        for arm in range(self.n_arms):
            count, mean = self._arm_stats(arm)
            bonus = math.sqrt(2.0 * log_t / max(count, 1e-9))
            scores.append(mean + bonus)
        return max(range(self.n_arms), key=lambda a: scores[a])

    def update(self, arm: int, reward: float) -> None:
        self.total_pulls += 1
        if self.mode == "discounted":
            self.discounted_counts = [c * self.gamma for c in self.discounted_counts]
            self.discounted_rewards = [r * self.gamma for r in self.discounted_rewards]
            self.discounted_counts[arm] += 1.0
            self.discounted_rewards[arm] += reward
            return

        self.windows[arm].append(reward)
