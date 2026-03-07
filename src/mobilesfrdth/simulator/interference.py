"""Modèles d'interférence et de décision de succès de trame."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

SNR_THRESHOLDS_DB: dict[int, float] = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}


@dataclass(frozen=True)
class InterferenceConfig:
    """Configuration SNIR/SINR."""

    snir_enabled: bool = False
    inter_sf_enabled: bool = True
    noise_floor_dbm: float = -120.0
    snr_thresholds_db: Mapping[int, float] = field(default_factory=lambda: dict(SNR_THRESHOLDS_DB))
    alpha_matrix: Mapping[int, Mapping[int, float]] | None = None

    def alpha(self, sf_interferer: int, sf_signal: int) -> float:
        """Retourne le coefficient d'interférence alpha(SF_i, SF_s)."""

        if sf_interferer == sf_signal:
            return 1.0
        if not self.inter_sf_enabled:
            return 0.0
        if self.alpha_matrix and sf_interferer in self.alpha_matrix:
            return float(self.alpha_matrix[sf_interferer].get(sf_signal, 0.0))
        return 0.1


def dbm_to_mw(power_dbm: float) -> float:
    return 10.0 ** (power_dbm / 10.0)


def mw_to_db(value_mw: float) -> float:
    return 10.0 * math.log10(max(value_mw, 1e-18))


def snr_db(signal_dbm: float, noise_floor_dbm: float) -> float:
    return signal_dbm - noise_floor_dbm


def sinr_db(
    signal_dbm: float,
    *,
    signal_sf: int,
    interferers: list[tuple[float, int]],
    cfg: InterferenceConfig,
) -> float:
    """Calcule ``SINR = Pr_signal / (N0 + Σ Pr_i * alpha(SF_i,SF_s))`` en dB."""

    signal_mw = dbm_to_mw(signal_dbm)
    noise_mw = dbm_to_mw(cfg.noise_floor_dbm)
    interf_mw = 0.0
    for power_i_dbm, sf_i in interferers:
        interf_mw += dbm_to_mw(power_i_dbm) * cfg.alpha(sf_interferer=sf_i, sf_signal=signal_sf)
    return mw_to_db(signal_mw / max(noise_mw + interf_mw, 1e-18))


def transmission_success(
    signal_dbm: float,
    *,
    signal_sf: int,
    interferers: list[tuple[float, int]],
    cfg: InterferenceConfig,
) -> tuple[bool, float]:
    """Décide le succès d'une trame.

    * SNIR_OFF: compare le SNR au seuil de SF.
    * SNIR_ON: compare le SINR au seuil de SF.
    """

    threshold = float(cfg.snr_thresholds_db.get(signal_sf, -20.0))
    if not cfg.snir_enabled:
        metric = snr_db(signal_dbm, cfg.noise_floor_dbm)
        return metric >= threshold, metric

    metric = sinr_db(signal_dbm, signal_sf=signal_sf, interferers=interferers, cfg=cfg)
    return metric >= threshold, metric
