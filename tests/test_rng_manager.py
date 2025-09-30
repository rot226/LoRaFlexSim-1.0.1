import numpy as np
import pytest

from traffic.rng_manager import (
    RngManager,
    activate_global_hooks,
    deactivate_global_hooks,
    UncontrolledRandomError,
)


def test_rng_seed_determinism():
    mgr1 = RngManager(123)
    mgr2 = RngManager(123)
    rng1 = mgr1.get_stream("traffic", 1)
    rng2 = mgr2.get_stream("traffic", 1)
    assert rng1.random() == rng2.random()


def test_global_hooks_block_unmanaged_rng():
    activate_global_hooks()
    try:
        with pytest.raises(UncontrolledRandomError):
            np.random.Generator(np.random.MT19937()).random()
    finally:
        deactivate_global_hooks()


def test_global_hooks_allow_registered_rng():
    mgr = RngManager(42)
    activate_global_hooks()
    try:
        rng = mgr.get_stream("foo")
        rng.random()
    finally:
        deactivate_global_hooks()
