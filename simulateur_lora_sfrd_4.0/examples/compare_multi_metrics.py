"""Exemple de comparaison multi-critères avec un export FLoRa."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_flora_report import generate_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare run.py results with FLoRa reference")
    parser.add_argument("flora_csv", help="CSV exporté depuis FLoRa")
    parser.add_argument("sim_csv", help="CSV généré par run.py")
    parser.add_argument(
        "--output-prefix",
        default="comparison",
        help="Préfixe des fichiers de sortie (par défaut: comparison)",
    )
    args = parser.parse_args()
    generate_report(args.flora_csv, args.sim_csv, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
