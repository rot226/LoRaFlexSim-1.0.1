"""Point d'entrée pour l'étape 1."""

from pathlib import Path

from article_c.common.csv_io import write_rows
from article_c.common.utils import ensure_dir
from article_c.step1.simulate_step1 import run_simulation


def main() -> None:
    result = run_simulation()
    results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(results_dir)
    output_path = results_dir / "step1_summary.csv"
    write_rows(output_path, ["sent", "received", "pdr"], [[result.sent, result.received, result.pdr]])


if __name__ == "__main__":
    main()
