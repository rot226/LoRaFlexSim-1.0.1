from __future__ import annotations

import random

from article_c.step2.bandit_ucb1 import BanditUCB1


def test_step2_ucb1_prefere_progressivement_le_sf_optimal() -> None:
    """Vérifie que UCB1 converge vers le SF ayant la meilleure récompense attendue."""
    sf_values = [7, 8, 9, 10]
    success_probabilities = [0.95, 0.55, 0.35, 0.10]
    optimal_sf_index = 0

    random.seed(2024)
    bandit = BanditUCB1(
        n_arms=len(sf_values),
        warmup_rounds=len(sf_values),
        epsilon_min=0.0,
    )

    selections: list[int] = []
    n_rounds = 600
    for _ in range(n_rounds):
        selected_index = bandit.select_arm()
        selections.append(selected_index)

        reward = 1.0 if random.random() < success_probabilities[selected_index] else 0.0
        bandit.update(selected_index, reward)

    early_window = selections[:120]
    late_window = selections[-120:]

    early_optimal_probability = early_window.count(optimal_sf_index) / len(early_window)
    late_optimal_probability = late_window.count(optimal_sf_index) / len(late_window)

    assert late_optimal_probability > early_optimal_probability + 0.20
    assert late_optimal_probability >= 0.90
