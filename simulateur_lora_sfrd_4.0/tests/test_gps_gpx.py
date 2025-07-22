import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.gps_mobility import GPSTraceMobility  # noqa: E402


def test_gpx_trace(tmp_path):
    gpx_file = tmp_path / "trace.gpx"
    gpx_file.write_text(
        """<gpx version='1.1' creator='test'>\n"
        "<trk><trkseg>\n"
        "<trkpt lat='0' lon='0'><time>2020-01-01T00:00:00Z</time></trkpt>\n"
        "<trkpt lat='10' lon='20'><time>2020-01-01T00:10:00Z</time></trkpt>\n"
        "</trkseg></trk></gpx>\n"""
    )
    mob = GPSTraceMobility(str(gpx_file))
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    mob.assign(node)
    mob.move(node, 300.0)
    assert node.x == pytest.approx(10.0)
    assert node.y == pytest.approx(5.0)
