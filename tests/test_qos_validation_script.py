import json
from pathlib import Path

from scripts import validate_qos_against_reference as validator


DATA_DIR = Path(__file__).resolve().parent / "data"
SERIES_PATH = DATA_DIR / "qos_validation_series.json"
REFERENCE_PATH = DATA_DIR / "qos_validation_reference.json"


def test_validate_qos_against_reference_passes(tmp_path, capsys):
    exit_code = validator.main(
        [
            "--series",
            str(SERIES_PATH),
            "--reference",
            str(REFERENCE_PATH),
            "--tolerance",
            "0.05",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "conformes" in captured.out


def test_validate_qos_against_reference_detects_cluster_gap(tmp_path):
    payload = json.loads(SERIES_PATH.read_text(encoding="utf8"))
    payload["entries"][0]["cluster_der_ratio"]["1"] = 0.80
    altered_series = tmp_path / "series.json"
    altered_series.write_text(json.dumps(payload), encoding="utf8")
    exit_code = validator.main(
        [
            "--series",
            str(altered_series),
            "--reference",
            str(REFERENCE_PATH),
            "--tolerance",
            "0.05",
        ]
    )
    assert exit_code == 1
