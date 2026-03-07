"""ADR legacy simplifié basé marge SNR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdrLegacyConfig:
    target_margin_db: float = 10.0
    min_sf: int = 7
    max_sf: int = 12


def recommend_sf(current_sf: int, snr_db: float, cfg: AdrLegacyConfig) -> int:
    """Ajuste le SF via une marge SNR crédible et simple."""

    new_sf = current_sf
    if snr_db > cfg.target_margin_db + 3.0:
        new_sf -= 1
    elif snr_db < cfg.target_margin_db - 3.0:
        new_sf += 1
    return max(cfg.min_sf, min(cfg.max_sf, new_sf))
