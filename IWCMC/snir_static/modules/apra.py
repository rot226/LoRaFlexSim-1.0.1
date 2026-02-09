"""Wrapper APRA pour les campagnes SNIR statiques."""

from __future__ import annotations

from loraflexsim.launcher.qos import QoSManager


def apply(simulator, manager: QoSManager | None = None) -> QoSManager:
    """Applique APRA-like sur ``simulator`` et retourne le gestionnaire."""

    qos_manager = manager or QoSManager()
    qos_manager.apply(simulator, "APRA-like")
    return qos_manager
