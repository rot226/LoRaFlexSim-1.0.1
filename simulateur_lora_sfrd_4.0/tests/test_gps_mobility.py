import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.gps_mobility import GPSTraceMobility, MultiGPSTraceMobility  # noqa: E402


def test_gps_trace_interpolation(tmp_path):
    trace_file = tmp_path / "trace.csv"
    trace_file.write_text("0,0,0\n10,10,0\n")
    mob = GPSTraceMobility(str(trace_file))
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    mob.assign(node)
    mob.move(node, 5.0)
    assert node.x == pytest.approx(5.0)
    assert node.y == pytest.approx(0.0)


def test_multi_gps_trace(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t1.csv").write_text("0,0,0\n10,10,0\n")
    (traces / "t2.csv").write_text("0,0,0\n10,0,10\n")
    mob = MultiGPSTraceMobility(str(traces))
    n1 = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    n2 = Node(2, 0.0, 0.0, 7, 14.0, channel=Channel())
    mob.assign(n1)
    mob.assign(n2)
    mob.move(n1, 5.0)
    mob.move(n2, 5.0)
    assert n1.x == pytest.approx(5.0)
    assert n2.y == pytest.approx(5.0)
