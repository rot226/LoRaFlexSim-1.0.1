"""Gestion centralisée des graines aléatoires pour mobile-sfrd."""

from __future__ import annotations

import hashlib

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Initialise les générateurs globaux et retourne un générateur NumPy.

    Args:
        seed: Graine entière utilisée pour initialiser l'aléatoire.

    Returns:
        Un ``numpy.random.Generator`` initialisé avec la graine fournie.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)


def spawn_rng(seed: int, stream_name: str) -> np.random.Generator:
    """Crée un flux RNG reproductible indépendant du flux principal.

    Le ``stream_name`` est haché puis combiné à ``seed`` pour garantir
    une séparation déterministe des streams (utile pour les figures/scénarios).
    """
    stream_hash = hashlib.sha256(stream_name.encode("utf-8")).digest()
    stream_offset = int.from_bytes(stream_hash[:8], byteorder="big", signed=False)
    mixed_seed = (seed + stream_offset) % (2**63 - 1)
    return np.random.default_rng(mixed_seed)
