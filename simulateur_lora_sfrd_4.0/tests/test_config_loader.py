import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.config_loader import load_config  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402


def test_load_config_parses_nodes_and_gateways(tmp_path):
    cfg = tmp_path / "net.ini"
    cfg.write_text(
        """[gateways]\n"""
        "gw0 = 0,0\n"
        "[nodes]\n"
        "n0 = 1,2,7,14\n"
        "n1 = 3,4\n"
    )
    nodes, gws = load_config(cfg)
    assert len(gws) == 1 and gws[0]["x"] == 0
    assert len(nodes) == 2 and nodes[1]["y"] == 4

    sim = Simulator(config_file=str(cfg))
    assert len(sim.nodes) == 2
    assert len(sim.gateways) == 1
    assert sim.nodes[0].x == 1
    assert sim.nodes[0].sf == 7
