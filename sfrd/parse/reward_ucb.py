"""Calcul/utilitaires de reward UCB."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class UCBEpisodeConfig:
    mode: str = "packets"
    packet_window: int = 1
    time_window_s: float = 60.0


@dataclass(frozen=True)
class UCBConfig:
    lambda_e: float = 0.5
    exploration_coefficient: float = 2.0
    reward_window: int = 20
    episode: UCBEpisodeConfig = UCBEpisodeConfig()


DEFAULT_UCB_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ucb_config.json"


def load_ucb_config(path: str | Path | None = None) -> UCBConfig:
    """Charge la configuration UCB externe avec fallback sur des valeurs sûres."""

    cfg_path = Path(path) if path is not None else DEFAULT_UCB_CONFIG_PATH
    if not cfg_path.is_file():
        return UCBConfig()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    episode_payload = payload.get("episode", {}) if isinstance(payload, dict) else {}
    if not isinstance(episode_payload, dict):
        episode_payload = {}

    mode = str(episode_payload.get("mode", "packets")).strip().lower()
    if mode not in {"packets", "time"}:
        mode = "packets"

    return UCBConfig(
        lambda_e=float(payload.get("lambda_E", 0.5)),
        exploration_coefficient=float(payload.get("exploration_coefficient", 2.0)),
        reward_window=max(1, int(payload.get("reward_window", 20))),
        episode=UCBEpisodeConfig(
            mode=mode,
            packet_window=max(1, int(episode_payload.get("packet_window", 1))),
            time_window_s=max(0.001, float(episode_payload.get("time_window_s", 60.0))),
        ),
    )

UCB_HISTORY_HEADERS = [
    "episode",
    "reward_raw",
    "reward_normalized",
    "chosen_sf",
    "success_rate",
    "bitrate_norm",
    "energy_norm",
]


def collect_ucb_history(simulator: Any) -> list[dict[str, Any]]:
    """Retourne l'historique UCB enregistré par le simulateur."""

    history = getattr(simulator, "ucb_history", None)
    if not isinstance(history, list):
        return []
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        payload["episode"] = int(payload.get("episode", index))
        normalized_rows.append(payload)
    return normalized_rows


def export_ucb_history_csv(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Écrit un CSV d'historique UCB dédié au run courant."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UCB_HISTORY_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in UCB_HISTORY_HEADERS})
    return path


def learning_curve_from_history(rows: Iterable[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Construit la courbe d'apprentissage UCB (episode/reward normalisée)."""

    curve: list[dict[str, float | int]] = []
    for index, row in enumerate(rows, start=1):
        try:
            episode = int(row.get("episode", index))
            reward = float(row.get("reward_normalized", 0.0))
        except (TypeError, ValueError):
            continue
        curve.append(
            {
                "episode": max(1, episode),
                "reward": reward,
            }
        )
    return curve


def aggregate_learning_curves(
    curves: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, float | int]]:
    """Aligne les runs par numéro d'épisode et moyenne simplement par épisode."""

    rewards_by_episode: dict[int, list[float]] = defaultdict(list)
    for curve in curves:
        for row in curve:
            try:
                episode = int(row.get("episode", 0))
                reward = float(row.get("reward", 0.0))
            except (TypeError, ValueError):
                continue
            if episode < 1:
                continue
            rewards_by_episode[episode].append(reward)

    return [
        {"episode": episode, "reward": sum(values) / len(values)}
        for episode, values in sorted(rewards_by_episode.items())
        if values
    ]
