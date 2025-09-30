import sys
import pathlib


def test_run_mobility_multichannel_script(tmp_path, monkeypatch):
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    # Temporarily remove repository and stubs paths to load real dependencies
    original_path = sys.path.copy()
    stub_numpy = sys.modules.get("numpy")
    stub_numpy_random = sys.modules.get("numpy.random")
    stub_pandas = sys.modules.get("pandas")
    if '' in sys.path:
        sys.path.remove('')
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    stubs_dir = repo_root / "tests" / "stubs"
    if str(stubs_dir) in sys.path:
        sys.path.remove(str(stubs_dir))
    sys.modules.pop("numpy", None)
    sys.modules.pop("numpy.random", None)
    sys.modules.pop("pandas", None)

    import numpy  # noqa: F401
    import pandas  # noqa: F401

    from scripts import run_mobility_multichannel

    monkeypatch.setattr(run_mobility_multichannel, "RESULTS_DIR", tmp_path)

    def fake_run_scenario(*args, **kwargs):
        return {
            "delivered": 80,
            "collisions": 20,
            "energy_nodes_J": 10.0,
            "avg_delay_s": 1.5,
            "avg_sf": 7.0,
        }

    monkeypatch.setattr(run_mobility_multichannel, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(run_mobility_multichannel, "range", lambda n: [0], raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_mobility_multichannel.py",
        "--replicates",
        "5",
    ])

    run_mobility_multichannel.main()

    csv_path = tmp_path / "mobility_multichannel.csv"
    assert csv_path.is_file()

    df = pandas.read_csv(csv_path)
    expected_cols = {
        "nodes",
        "channels",
        "pdr_mean",
        "pdr_std",
        "collision_rate_mean",
        "collision_rate_std",
        "avg_delay_s_mean",
        "avg_delay_s_std",
        "energy_per_node_mean",
        "energy_per_node_std",
        "avg_sf_mean",
        "avg_sf_std",
    }
    assert expected_cols.issubset(df.columns)

    row = df[df["scenario"] == "n200_c1_mobile"].iloc[0]
    assert row["nodes"] == 200
    assert row["channels"] == 1
    assert row["pdr_mean"] == 80.0
    assert bool(row["mobility"]) is True
    assert row["speed"] == 5.0

    static_row = df[df["scenario"] == "n50_c1_static"].iloc[0]
    assert not static_row["mobility"]
    assert static_row["speed"] == 0.0

    csv_path.unlink()

    # Restore stubbed modules and paths
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
