import pytest

from loraflexsim.launcher.propagation_cache import PropagationCache


def test_channel_path_loss_uses_cache():
    cache = PropagationCache(resolution=0.5, max_entries=16)
    calls = {"count": 0}

    def compute() -> float:
        calls["count"] += 1
        return 42.0

    assert cache.get(1_200.0, compute) == 42.0
    assert cache.get(1_200.2, compute) == 42.0
    assert calls["count"] == 1
    cache.clear()
    assert cache.get(1_200.0, compute) == 42.0
    assert calls["count"] == 2


def test_propagation_cache_validation():
    with pytest.raises(ValueError):
        PropagationCache(resolution=0.0)
