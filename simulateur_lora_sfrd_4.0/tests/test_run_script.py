import sys
from pathlib import Path

# Allow importing the VERSION_4 package from the repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import VERSION_4.run as run


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
