import sys
import pathlib


def test_battery_tracking_script(tmp_path, monkeypatch):
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    # Import real numpy/pandas by temporarily removing the repository and
    # ``numpy_stub`` paths so that the genuine dependencies from the
    # environment are used.
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

    from scripts import run_battery_tracking, plot_battery_tracking

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(run_battery_tracking, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_battery_tracking.py",
        "--nodes",
        "1",
        "--packets",
        "1",
        "--seed",
        "1",
    ])
    run_battery_tracking.main()
    csv_path = tmp_path / "battery_tracking.csv"
    assert csv_path.is_file()

    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(plot_battery_tracking, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(plot_battery_tracking, "FIGURES_DIR", figures_dir)
    plot_battery_tracking.main()
    for ext in ("png", "jpg", "eps"):
        path = figures_dir / f"battery_tracking.{ext}"
        assert path.is_file()
        path.unlink()
    figures_dir.rmdir()
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
