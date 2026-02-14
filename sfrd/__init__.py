"""Pont d'intégration SFRD vers LoRaFlexSim."""

from __future__ import annotations

from typing import Any

__all__ = ["run_campaign"]


def __getattr__(name: str) -> Any:
    if name == "run_campaign":
        from .cli import run_campaign as _run_campaign

        return _run_campaign
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
