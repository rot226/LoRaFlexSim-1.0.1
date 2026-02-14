from __future__ import annotations

from article_c.step1.plots.plot_S5 import _as_bool, _extract_aggregated_pdr_groups


def test_as_bool_supports_mixed_types() -> None:
    cases = [
        (True, True),
        (False, False),
        (None, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("1", True),
        ("  true ", True),
        ("yes", True),
        ("vrai", True),
        ("oui", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("non", False),
        ("faux", False),
        ("off", False),
        ("", False),
        ([], False),
    ]

    for raw_value, expected in cases:
        assert _as_bool(raw_value) is expected


def test_extract_aggregated_groups_filters_mixra_opt_fallback_for_mixed_types() -> None:
    rows = [
        {
            "network_size": 1280,
            "algo": "mixra_opt",
            "snir_mode": "snir_on",
            "aggregated_pdr": 0.92,
            "mixra_opt_fallback": True,
        },
        {
            "network_size": 1280,
            "algo": "mixra_opt",
            "snir_mode": "snir_on",
            "aggregated_pdr": 0.90,
            "mixra_opt_fallback": "1",
        },
        {
            "network_size": 1280,
            "algo": "mixra_opt",
            "snir_mode": "snir_off",
            "aggregated_pdr": 0.88,
            "mixra_opt_fallback": "false",
        },
        {
            "network_size": 1280,
            "algo": "mixra_opt",
            "snir_mode": "snir_on",
            "aggregated_pdr": 0.86,
            "mixra_opt_fallback": 0,
        },
    ]

    values_by_size = _extract_aggregated_pdr_groups(rows)
    groups = values_by_size[1280]

    assert ("mixra_opt", True, "snir_on") not in groups
    assert groups[("mixra_opt", False, "snir_off")] == [0.88]
    assert groups[("mixra_opt", False, "snir_on")] == [0.86]
