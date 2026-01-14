"""Simulation de l'étape 2 (placeholder)."""

from dataclasses import dataclass

from article_c.step2.bandit_ucb1 import UCB1


@dataclass
class Step2Result:
    total_reward: float


def run_simulation(rounds: int = 10, n_arms: int = 3) -> Step2Result:
    bandit = UCB1(n_arms)
    total_reward = 0.0
    for _ in range(rounds):
        arm = bandit.select_arm()
        reward = 1.0 if arm == 0 else 0.5
        bandit.update(arm, reward)
        total_reward += reward
    return Step2Result(total_reward=total_reward)
