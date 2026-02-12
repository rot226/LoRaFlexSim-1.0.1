"""Inspecte le contrat d'agrégation de résultats (`aggregates` et `by_size`).

Vérifie :
- existence des tailles attendues (`by_size/size_<N>/aggregated_results.csv`)
- présence de l'agrégat global (`aggregates/aggregated_results.csv`)
- cohérence du nombre total de lignes entre agrégats par taille et global
"""

from __future__ import annotations

import argparse
import csv
import sys
from importlib.util import find_spec
from pathlib import Path

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from article_c.common.config import BASE_DIR

EXPECTED_SIZES: tuple[int, ...] = (80, 160, 320, 640, 1280)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspecte les agrégats results/aggregates et results/by_size."
    )
    parser.add_argument(
        "--step",
        choices=("step1", "step2"),
        default="step1",
        help="Étape à inspecter (par défaut: step1).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Chemin explicite vers le dossier results (sinon: article_c/<step>/results).",
    )
    return parser


def _count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def main() -> int:
    args = _build_parser().parse_args()

    results_dir = args.results_dir or (BASE_DIR / args.step / "results")
    print(f"Dossier inspecté: {results_dir}")

    if not results_dir.exists():
        print("ERREUR: dossier results introuvable.")
        return 1

    failures: list[str] = []
    missing_sizes: list[str] = []
    rows_by_size: dict[int, int] = {size: 0 for size in EXPECTED_SIZES}
    by_size_total = 0

    for size in EXPECTED_SIZES:
        size_csv = results_dir / "by_size" / f"size_{size}" / "aggregated_results.csv"
        if not size_csv.exists():
            missing_sizes.append(f"size_{size}")
            continue
        try:
            rows = _count_data_rows(size_csv)
        except Exception as exc:
            failures.append(f"{size_csv.relative_to(results_dir)}: lecture impossible ({exc})")
            continue
        rows_by_size[size] = rows
        by_size_total += rows
        if rows == 0:
            failures.append(f"size_{size}: aggregated_results.csv sans ligne de données")

    global_csv = results_dir / "aggregates" / "aggregated_results.csv"
    global_rows = 0
    if not global_csv.exists():
        failures.append("aggregates/aggregated_results.csv absent")
    else:
        try:
            global_rows = _count_data_rows(global_csv)
        except Exception as exc:
            failures.append(f"aggregates/aggregated_results.csv: lecture impossible ({exc})")

    print("\nRésumé lignes de données par taille:")
    for size in EXPECTED_SIZES:
        print(f"- size_{size}: {rows_by_size[size]} ligne(s)")
    print(f"- global: {global_rows} ligne(s)")

    if global_csv.exists() and by_size_total != global_rows:
        failures.append(
            "Incohérence contrat agrégé: "
            f"somme(by_size)={by_size_total} != global={global_rows}"
        )

    if missing_sizes:
        failures.append(f"Tailles manquantes: {', '.join(missing_sizes)}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
    else:
        print("\nPASS")

    if missing_sizes:
        return 2

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
