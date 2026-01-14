"""Exécute toutes les étapes de l'article C."""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path) -> None:
    subprocess.check_call([sys.executable, str(script_path)])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    run_script(base_dir / "step1" / "run_step1.py")
    run_script(base_dir / "step2" / "run_step2.py")


if __name__ == "__main__":
    main()
