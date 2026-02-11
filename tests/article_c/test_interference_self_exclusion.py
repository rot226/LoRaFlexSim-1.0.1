from __future__ import annotations

from article_c.common.interference import Signal, evaluate_reception


def test_single_transmitter_does_not_count_itself_as_interference() -> None:
    target = Signal(
        rssi_dbm=-110.0,
        sf=7,
        channel_hz=868100000,
        start_time_s=0.0,
        end_time_s=1.0,
    )

    outcome = evaluate_reception(
        target,
        [target],
        sensitivity_dbm=-120.0,
        snir_enabled=True,
        snir_threshold_db=-20.0,
        snir_threshold_min_db=-20.0,
        snir_threshold_max_db=20.0,
    )

    assert outcome.success is True
    assert outcome.outage is False
    assert outcome.co_sf_collisions == 0
    assert outcome.interference_dbm == float("-inf")
