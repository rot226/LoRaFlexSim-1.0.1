import pathlib
import sys


import pytest


def test_battery_tracking_stops_on_depletion(tmp_path, monkeypatch, capsys):
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    original_path = sys.path.copy()
    stub_numpy = sys.modules.get("numpy")
    stub_numpy_random = sys.modules.get("numpy.random")
    stub_pandas = sys.modules.get("pandas")
    if "" in sys.path:
        sys.path.remove("")
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    stubs_dir = repo_root / "tests" / "stubs"
    if str(stubs_dir) in sys.path:
        sys.path.remove(str(stubs_dir))

    sys.modules.pop("numpy", None)
    sys.modules.pop("numpy.random", None)
    sys.modules.pop("pandas", None)

    import numpy  # noqa: F401
    import pandas as pd

    from scripts import run_battery_tracking

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(run_battery_tracking, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_battery_tracking.py",
            "--nodes",
            "1",
            "--packets",
            "5",
            "--seed",
            "1",
            "--battery-capacity-j",
            "0.001",
            "--stop-on-depletion",
        ],
    )

    run_battery_tracking.main()

    captured = capsys.readouterr()
    assert "Arrêt: batteries épuisées" in captured.out

    csv_path = tmp_path / "battery_tracking.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    assert pytest.approx(0.0, abs=1e-12) == last_row["energy_j"]
    assert not bool(last_row["alive"])

    # Conserver un nombre raisonnable d'enregistrements pour confirmer l'arrêt anticipé
    assert len(df) < 50

    csv_path.unlink()

    sys.path = original_path
    sys.modules.pop("numpy", None)
    sys.modules.pop("numpy.random", None)
    sys.modules.pop("pandas", None)
    if stub_numpy is not None:
        sys.modules["numpy"] = stub_numpy
    if stub_numpy_random is not None:
        sys.modules["numpy.random"] = stub_numpy_random
    if stub_pandas is not None:
        sys.modules["pandas"] = stub_pandas
