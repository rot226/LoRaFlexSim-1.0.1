"""Couplage ADR (SF) + MixRA (puissance)."""

from __future__ import annotations

from dataclasses import dataclass

from .adr_legacy import AdrLegacyConfig, recommend_sf
from .mixra import MixRaConfig, adjust_tx_power


@dataclass(frozen=True)
class AdrMixRaConfig:
    adr: AdrLegacyConfig = AdrLegacyConfig()
    mixra: MixRaConfig = MixRaConfig()


def adapt_link(
    *,
    current_sf: int,
    current_tx_power_dbm: float,
    snr_db: float,
    pdr_estimate: float,
    latency_estimate_s: float,
    cfg: AdrMixRaConfig,
) -> tuple[int, float]:
    """Retourne ``(sf, tx_power_dbm)`` après adaptation conjointe."""

    sf = recommend_sf(current_sf=current_sf, snr_db=snr_db, cfg=cfg.adr)
    tx = adjust_tx_power(
        current_tx_power_dbm=current_tx_power_dbm,
        pdr_estimate=pdr_estimate,
        latency_estimate_s=latency_estimate_s,
        cfg=cfg.mixra,
    )
    return sf, tx
