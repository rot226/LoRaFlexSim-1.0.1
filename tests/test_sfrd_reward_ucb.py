from __future__ import annotations

import json

from sfrd.cli import run_campaign
from sfrd.parse.reward_ucb import (
    aggregate_learning_curves,
    learning_curve_from_history,
    load_ucb_config,
)


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


def test_load_ucb_config_reads_external_json(tmp_path) -> None:
    config_path = tmp_path / "ucb.json"
    config_path.write_text(
        json.dumps(
            {
                "lambda_E": 0.35,
                "exploration_coefficient": 1.4,
                "reward_window": 12,
                "episode": {"mode": "time", "packet_window": 4, "time_window_s": 45.0},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_ucb_config(config_path)

    assert cfg.lambda_e == 0.35
    assert cfg.exploration_coefficient == 1.4
    assert cfg.reward_window == 12
    assert cfg.episode.mode == "time"
    assert cfg.episode.packet_window == 4
    assert cfg.episode.time_window_s == 45.0


def test_run_campaign_applies_packet_episode_window(tmp_path) -> None:
    config_path = tmp_path / "ucb.json"
    config_path.write_text(
        json.dumps(
            {
                "lambda_E": 0.5,
                "exploration_coefficient": 1.0,
                "reward_window": 10,
                "episode": {"mode": "packets", "packet_window": 3, "time_window_s": 60.0},
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "run"
    run_campaign(
        network_size=6,
        algorithm="ucb",
        snir_mode="snir_on",
        seed=7,
        warmup_s=0.0,
        output_dir=output_dir,
        ucb_config_path=config_path,
    )

    ucb_history_path = output_dir / "ucb_history.csv"
    lines = [line for line in ucb_history_path.read_text(encoding="utf-8").splitlines() if line]
    episodes = [int(line.split(",")[0]) for line in lines[1:]]

    assert episodes
    assert episodes[0] == 1
    assert max(episodes) == (len(episodes) - 1) // 3 + 1
