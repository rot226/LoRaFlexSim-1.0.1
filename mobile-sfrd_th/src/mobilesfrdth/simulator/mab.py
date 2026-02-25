"""Compatibilité: expose les algorithmes MAB en sous-package."""

from .mab import UCB1, UCBForget

__all__ = ["UCB1", "UCBForget"]
