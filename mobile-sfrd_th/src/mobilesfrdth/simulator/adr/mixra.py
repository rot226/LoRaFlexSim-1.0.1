"""MixRA simplifié orienté QoS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixRaConfig:
    pdr_target: float = 0.9
    latency_target_s: float = 5.0
    step_db: float = 1.0
    min_tx_power_dbm: float = 2.0
    max_tx_power_dbm: float = 14.0


def adjust_tx_power(
    current_tx_power_dbm: float,
    *,
    pdr_estimate: float,
    latency_estimate_s: float,
    cfg: MixRaConfig,
) -> float:
    """Pilotage puissance avec priorité à la robustesse QoS."""

    tx = current_tx_power_dbm
    degraded = pdr_estimate < cfg.pdr_target or latency_estimate_s > cfg.latency_target_s
    if degraded:
        tx += cfg.step_db
    else:
        tx -= cfg.step_db
    return max(cfg.min_tx_power_dbm, min(cfg.max_tx_power_dbm, tx))
