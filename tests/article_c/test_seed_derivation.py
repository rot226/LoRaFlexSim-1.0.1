from __future__ import annotations

from article_c.common.utils import derive_run_seed


def test_derive_run_seed_is_stable_for_same_tuple() -> None:
    seed_a = derive_run_seed(
        seeds_base=123,
        network_size=100,
        replication=2,
        algo="mixra_opt",
        snir_mode="snir_on",
    )
    seed_b = derive_run_seed(
        seeds_base=123,
        network_size=100,
        replication=2,
        algo="mixra_opt",
        snir_mode="snir_on",
    )

    assert seed_a == seed_b


def test_derive_run_seed_changes_when_tuple_changes() -> None:
    base_kwargs = {
        "seeds_base": 123,
        "network_size": 100,
        "replication": 2,
        "algo": "mixra_opt",
        "snir_mode": "snir_on",
    }
    seed_ref = derive_run_seed(**base_kwargs)

    variations = [
        {**base_kwargs, "seeds_base": 124},
        {**base_kwargs, "network_size": 101},
        {**base_kwargs, "replication": 3},
        {**base_kwargs, "algo": "adr"},
        {**base_kwargs, "snir_mode": "snir_off"},
    ]

    for variant in variations:
        assert derive_run_seed(**variant) != seed_ref
