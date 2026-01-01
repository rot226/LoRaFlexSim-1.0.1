from __future__ import annotations

import pytest

from scripts import run_step1_experiments


def test_ensure_collisions_snir_requires_field():
    with pytest.raises(ValueError, match="collisions_snir"):
        run_step1_experiments._ensure_collisions_snir({})


def test_ensure_collisions_snir_accepts_field():
    run_step1_experiments._ensure_collisions_snir({"collisions_snir": 0})
