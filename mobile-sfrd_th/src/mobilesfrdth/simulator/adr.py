"""Compatibilité: expose les algorithmes ADR en sous-package."""

from .adr import adr_legacy, adr_mixra, mixra

__all__ = ["adr_legacy", "mixra", "adr_mixra"]
