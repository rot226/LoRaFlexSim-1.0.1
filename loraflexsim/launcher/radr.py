from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .simulator import Simulator


@dataclass
class RADRModel:
    """Minimal reinforcement learning model for RADR.

    The model maintains a table of cumulative rewards for state-action
    pairs.  It is intentionally lightweight and serves as a placeholder
    for a more advanced learning algorithm.
    """

    q_table: Dict[Tuple[Tuple, str], float] = field(default_factory=dict)

    def update(self, state: Tuple, action: str, reward: float) -> None:
        """Update the cumulative reward for a given state and action."""
        key = (state, action)
        self.q_table[key] = self.q_table.get(key, 0.0) + reward

    def best_action(self, state: Tuple) -> str:
        """Return the best known action for *state*.

        If no data is available for *state*, ``"keep"`` is returned.
        """
        candidates = {a: r for (s, a), r in self.q_table.items() if s == state}
        if not candidates:
            return "keep"
        return max(candidates, key=candidates.get)


def apply(sim: Simulator) -> None:
    """Configure the RADR ADR method using reinforcement learning.

    The network server exposes two callbacks:

    ``adr_action(snr, sf)``
        Choose the action (``"sf_up"``, ``"sf_down"`` or ``"keep"``)
        based on the observed SNR and current spreading factor ``sf``.

    ``adr_reward(snr, sf, action)``
        Update the model with the reward computed from the difference
        between the measured SNR and the required SNR for ``sf``.
    """

    sim.network_server.adr_method = "radr"
    model = RADRModel()
    sim.network_server.adr_model = model

    def action_callback(snr: float, sf: int) -> str:
        state = (round(snr), sf)
        action = model.best_action(state)
        # If no learned action exists, fall back to simple heuristics.
        if (state, action) not in model.q_table:
            required = Simulator.REQUIRED_SNR.get(sf, -20.0)
            if snr < required:
                action = "sf_up"
            elif snr > required + Simulator.MARGIN_DB:
                action = "sf_down"
            else:
                action = "keep"
        return action

    def reward_callback(snr: float, sf: int, action: str) -> None:
        state = (round(snr), sf)
        required = Simulator.REQUIRED_SNR.get(sf, -20.0)
        reward = snr - required
        model.update(state, action, reward)

    sim.network_server.adr_action = action_callback
    sim.network_server.adr_reward = reward_callback
