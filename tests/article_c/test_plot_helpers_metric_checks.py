from __future__ import annotations

import logging
import warnings

from article_c.common.plot_helpers import warn_metric_checks_by_group


def test_warn_metric_checks_by_group_single_network_size_no_warn_spam(caplog) -> None:
    rows = [
        {
            "network_size": 100,
            "algo": "adr",
            "snir_mode": "snir_on",
            "cluster": "all",
            "pdr": 0.91,
            "pdr_mean": 0.91,
        },
        {
            "network_size": 100,
            "algo": "adr",
            "snir_mode": "snir_off",
            "cluster": "all",
            "pdr": 0.84,
            "pdr_mean": 0.84,
        },
    ]

    with caplog.at_level(logging.INFO), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warn_metric_checks_by_group(
            rows,
            "pdr",
            x_key="network_size",
            label="S6/PDR",
            min_value=0.0,
            max_value=1.0,
            expected_monotonic="decreasing",
            group_keys=("algo", "snir_mode", "cluster"),
        )

    warn_logs = [record for record in caplog.records if "[METRIC-CHECK][WARN]" in record.message]
    skip_logs = [record for record in caplog.records if "[METRIC-CHECK][SKIP]" in record.message]

    assert not warn_logs
    assert len(skip_logs) == 2
    assert all("campagne partielle" in record.message for record in skip_logs)
    assert not captured
