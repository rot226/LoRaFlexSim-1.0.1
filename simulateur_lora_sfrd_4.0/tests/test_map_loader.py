import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.map_loader import load_map  # noqa: E402


def test_load_map_json_and_text(tmp_path):
    data = [[0, 1], [2, 3]]
    json_file = tmp_path / "map.json"
    json_file.write_text(json.dumps(data))
    assert load_map(json_file) == [[0.0, 1.0], [2.0, 3.0]]

    txt_file = tmp_path / "map.txt"
    txt_file.write_text("0 1\n2 3\n")
    assert load_map(txt_file) == [[0.0, 1.0], [2.0, 3.0]]
