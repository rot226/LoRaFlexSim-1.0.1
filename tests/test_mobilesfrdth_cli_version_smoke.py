import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_mobilesfrdth_unsupported_python_smoke() -> None:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    code = (
        "import sys; "
        "sys.version_info=(3,10,9,'final',0); "
        "from mobilesfrdth import cli; "
        "raise SystemExit(cli.main(['--help']))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Version Python non supportée" in completed.stderr
    assert "utiliser une version >=3.11 et <3.15" in completed.stderr
