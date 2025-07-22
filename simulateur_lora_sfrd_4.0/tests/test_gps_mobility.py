import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.gps_mobility import GPSTraceMobility  # noqa: E402


def test_gps_trace_interpolation(tmp_path):
    trace_file = tmp_path / "trace.csv"
    trace_file.write_text("0,0,0\n10,10,0\n")
    mob = GPSTraceMobility(str(trace_file))
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    mob.assign(node)
    mob.move(node, 5.0)
    assert node.x == pytest.approx(5.0)
    assert node.y == pytest.approx(0.0)
