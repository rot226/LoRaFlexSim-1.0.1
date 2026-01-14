"""Point d'entrée pour l'étape 2."""

from pathlib import Path

from article_c.common.csv_io import write_rows
from article_c.common.utils import ensure_dir
from article_c.step2.simulate_step2 import run_simulation


def main() -> None:
    result = run_simulation()
    results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(results_dir)
    output_path = results_dir / "step2_summary.csv"
    write_rows(output_path, ["total_reward"], [[result.total_reward]])


if __name__ == "__main__":
    main()
