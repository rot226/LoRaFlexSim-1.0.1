from __future__ import annotations

from sfrd.parse.reward_ucb import aggregate_learning_curves, learning_curve_from_history


def test_learning_curve_from_history_uses_normalized_reward_and_episode_start_at_1() -> None:
    history = [
        {"episode": 0, "reward_normalized": 0.25},
        {"episode": 2, "reward_normalized": 0.75},
    ]

    curve = learning_curve_from_history(history)

    assert curve == [
        {"episode": 1, "reward": 0.25},
        {"episode": 2, "reward": 0.75},
    ]


def test_aggregate_learning_curves_is_simple_mean_by_episode() -> None:
    curve_a = [
        {"episode": 1, "reward": 0.2},
        {"episode": 2, "reward": 0.8},
    ]
    curve_b = [
        {"episode": 1, "reward": 0.4},
        {"episode": 3, "reward": 0.6},
    ]

    aggregated = aggregate_learning_curves([curve_a, curve_b])

    assert aggregated == [
        {"episode": 1, "reward": 0.30000000000000004},
        {"episode": 2, "reward": 0.8},
        {"episode": 3, "reward": 0.6},
    ]
