"""Wrapper MixRA-Opt pour les campagnes SNIR statiques."""

from __future__ import annotations

from loraflexsim.launcher.qos import QoSManager


def apply(simulator, manager: QoSManager | None = None) -> QoSManager:
    """Applique MixRA-Opt sur ``simulator`` et retourne le gestionnaire."""

    qos_manager = manager or QoSManager()
    qos_manager.apply(simulator, "MixRA-Opt")
    return qos_manager
