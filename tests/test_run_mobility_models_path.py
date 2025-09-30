import sys
import json
from pathlib import Path

from scripts import run_mobility_models


def test_run_mobility_models_path(tmp_path, monkeypatch):
    # Prepare simple map file with all cells set to 0
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps([[0, 0], [0, 0]]))

    # Redirect results to temporary directory
    monkeypatch.setattr(run_mobility_models, "RESULTS_DIR", tmp_path)

    # Invoke the script for the path model using minimal parameters
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mobility_models.py",
            "--nodes",
            "1",
            "--packets",
            "1",
            "--seed",
            "1",
            "--model",
            "path",
            "--path-map",
            str(map_path),
        ],
    )
    run_mobility_models.main()

    csv_path = tmp_path / "mobility_models.csv"
    assert csv_path.is_file()

    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    header_first = lines[0].split(",")[0]
    row_first = lines[1].split(",")[0]
    # Ensure header and first row correspond to 'model' and 'path'
    assert f"{header_first},{row_first}" == "model,path"
