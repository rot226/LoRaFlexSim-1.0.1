import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.config_loader import load_config, write_flora_ini


def test_write_flora_ini_roundtrip(tmp_path):
    nodes = [
        {"x": 1.0, "y": 2.0, "sf": 7, "tx_power": 14.0},
        {"x": 3.0, "y": 4.0, "sf": 9, "tx_power": 12.0},
    ]
    gws = [{"x": 0.0, "y": 0.0}]
    ini = tmp_path / "scene.ini"
    write_flora_ini(nodes, gws, ini)
    loaded_nodes, loaded_gws = load_config(ini)
    assert loaded_gws == gws
    assert loaded_nodes == nodes
