"""Point d'entrée pour l'étape 2."""

from pathlib import Path

from article_c.common.utils import ensure_dir
from article_c.step2.simulate_step2 import run_simulation


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    ensure_dir(results_dir)
    run_simulation(output_dir=results_dir)


if __name__ == "__main__":
    main()
