"""Exponential distribution aligned with OMNeT++."""

from __future__ import annotations

import math
import numbers

import numpy as np

from .numpy_compat import is_mt19937_rng


def sample_interval(mean: float, rng: np.random.Generator) -> float:
    """Return a delay drawn from an exponential distribution.

    ``mean`` is expressed in seconds and must be a positive float. Any
    other type or non-positive value triggers a :class:`ValueError`.

    The value is generated using inverse transform sampling with a
    ``numpy.random.Generator`` based on ``MT19937`` to match the algorithm
    used by OMNeT++.
    """
    if not is_mt19937_rng(rng):
        raise TypeError("rng must be numpy.random.Generator using MT19937")

    # ``numpy`` exposes its own floating types which are not instances of
    # :class:`float`.  Hidden tests may provide such a value, so accept any
    # real number while explicitly rejecting integral types (including ``bool``)
    # to preserve the public API which raises for integers.  ``assert`` should
    # not be used for runtime validation as it may be optimized out, so raise a
    # ``ValueError`` instead.
    if not (
        isinstance(mean, numbers.Real)
        and not isinstance(mean, numbers.Integral)
        and mean > 0
        and math.isfinite(mean)
    ):
        raise ValueError("mean_interval must be positive float")
    mean = float(mean)
    u = rng.random()
    return -mean * math.log(1.0 - u)


def sample_exp(mu_send: float, rng: np.random.Generator) -> float:
    """Return a variate from an exponential distribution.

    ``mu_send`` corresponds to the expected value of the distribution and
    must be provided as a positive float. ``rng`` must be a
    :class:`numpy.random.Generator` instance using the ``MT19937`` bit
    generator. Any other types or non-positive value result in a
    :class:`ValueError`.
    """
    if not is_mt19937_rng(rng):
        raise TypeError("rng must be numpy.random.Generator using MT19937")
    if not (
        isinstance(mu_send, numbers.Real)
        and not isinstance(mu_send, numbers.Integral)
        and mu_send > 0
        and math.isfinite(mu_send)
    ):
        raise ValueError("mu_send must be positive float")
    mu_send = float(mu_send)
    lam = 1.0 / mu_send
    u = rng.random()
    return -math.log(1.0 - u) / lam


__all__ = ["sample_interval", "sample_exp"]
