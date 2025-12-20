import pytest

from loraflexsim.learning import LoRaSFSelectorUCB1, UCB1Bandit


def test_reward_nominal_case():
    selector = LoRaSFSelectorUCB1(
        success_weight=1.0,
        snir_margin_weight=0.5,
        snir_threshold_db=0.0,
    )

    reward = selector.reward_from_outcome(True, snir_db=2.0)

    assert reward == pytest.approx(1.5)


def test_reward_marginal_snir_penalty():
    selector = LoRaSFSelectorUCB1(
        success_weight=1.0,
        snir_margin_weight=0.5,
        snir_threshold_db=0.0,
    )

    reward = selector.reward_from_outcome(
        True, snir_db=0.1, marginal_snir_margin_db=0.5
    )

    assert reward == pytest.approx(0.85)


def test_reward_energy_and_collision_penalties():
    selector = LoRaSFSelectorUCB1(
        success_weight=1.0,
        energy_penalty_weight=0.6,
        collision_penalty=0.2,
        energy_normalization=2.0,
    )

    reward = selector.reward_from_outcome(
        True,
        energy_j=1.0,
        collision=True,
    )

    assert reward == pytest.approx(0.5)


def test_reward_with_fairness_component():
    selector = LoRaSFSelectorUCB1(
        success_weight=1.0,
        fairness_weight=0.4,
    )

    reward = selector.reward_from_outcome(
        False,
        fairness_index=0.75,
    )

    assert reward == pytest.approx(0.3)


def test_reward_normalization_with_expected_der():
    selector = LoRaSFSelectorUCB1(success_weight=1.0)

    reward = selector.reward_from_outcome(True, expected_der=2.0)

    assert reward == pytest.approx(0.5)


def test_ucb1_bandit_weighted_statistics():
    bandit = UCB1Bandit(n_arms=1, window_size=5, traffic_weighted_mean=True)

    bandit.update(0, reward=1.0, weight=2.0)
    bandit.update(0, reward=0.0, weight=1.0)
    bandit.update(0, reward=0.5, weight=3.0)

    expected_mean = (1.0 * 2.0 + 0.0 * 1.0 + 0.5 * 3.0) / 6.0
    expected_variance = (
        (2.0 * (1.0 - expected_mean) ** 2)
        + (1.0 * (0.0 - expected_mean) ** 2)
        + (3.0 * (0.5 - expected_mean) ** 2)
    ) / 6.0

    assert bandit.reward_window_mean[0] == pytest.approx(expected_mean)
    assert bandit.reward_window_variance[0] == pytest.approx(expected_variance)
