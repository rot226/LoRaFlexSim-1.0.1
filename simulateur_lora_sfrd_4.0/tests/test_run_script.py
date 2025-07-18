import sys
from pathlib import Path
import pytest

# Allow importing the VERSION_4 package from the repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import VERSION_4.run as run  # noqa: E402


def test_simulate_throughput_bps():
    delivered, collisions, pdr, energy, avg_delay, throughput = run.simulate(
        nodes=1,
        gateways=1,
        mode="Periodic",
        interval=1,
        steps=10,
        channels=1,
    )
    assert delivered == 10
    expected_throughput = delivered * run.PAYLOAD_SIZE * 8 / 10
    assert throughput == expected_throughput


def test_main_runs_multiple_times(monkeypatch):
    calls = []

    def fake_simulate(*args, **kwargs):
        calls.append(1)
        return (1, 2, 3, 4, 5, 6)

    monkeypatch.setattr(run, "simulate", fake_simulate)
    results, avg = run.main(["--runs", "3"])

    assert len(calls) == 3
    assert results == [(1, 2, 3, 4, 5, 6)] * 3
    assert avg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_simulate_invalid_arguments():
    with pytest.raises(ValueError):
        run.simulate(
            nodes=0,
            gateways=1,
            mode="Periodic",
            interval=1,
            steps=10,
            channels=1,
        )

    with pytest.raises(ValueError):
        run.simulate(
            nodes=1,
            gateways=1,
            mode="Periodic",
            interval=1,
            steps=10,
            channels=0,
        )

    # Ensure invalid steps values raise errors
    with pytest.raises(ValueError):
        run.simulate(
            nodes=1,
            gateways=1,
            mode="Periodic",
            interval=1,
            steps=0,
            channels=1,
        )

    with pytest.raises(ValueError):
        run.simulate(
            nodes=1,
            gateways=1,
            mode="Periodic",
            interval=1,
            steps=-5,
            channels=1,
        )


def test_avg_delay_not_zero():
    results, averages = run.main([
        "--nodes",
        "1",
        "--mode",
        "periodic",
        "--interval",
        "1",
        "--steps",
        "5",
    ])
    # avg_delay is the 5th item in the tuple returned by run.main
    assert results[0][4] != 0
    assert averages[4] != 0
