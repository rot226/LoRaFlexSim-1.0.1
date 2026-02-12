"""Inspecte l'arborescence de résultats nested (`size_*/rep_*`).

Vérifie :
- existence des tailles attendues (`size_80`, `size_160`, `size_320`, `size_640`, `size_1280`)
- présence des fichiers requis par réplication (`raw_*.csv`, `aggregated_results.csv`)
- nombre de lignes de données non nul par taille

Le script renvoie un code non-zéro si au moins une taille attendue est absente.
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
        description="Inspecte les résultats nested size_*/rep_*."
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
            next(reader)  # en-tête
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _iter_rep_dirs(size_dir: Path) -> list[Path]:
    return sorted(path for path in size_dir.iterdir() if path.is_dir() and path.name.startswith("rep_"))


def main() -> int:
    args = _build_parser().parse_args()

    results_dir = args.results_dir or (BASE_DIR / args.step / "results")
    print(f"Dossier inspecté: {results_dir}")

    if not results_dir.exists():
        print("ERREUR: dossier results introuvable.")
        return 1

    failures: list[str] = []
    missing_sizes: list[str] = []

    # 1) Existence des tailles attendues
    size_dirs: dict[int, Path] = {}
    for size in EXPECTED_SIZES:
        size_dir = results_dir / f"size_{size}"
        if not size_dir.exists() or not size_dir.is_dir():
            missing_sizes.append(size_dir.name)
            continue
        size_dirs[size] = size_dir

    if missing_sizes:
        failures.append(f"Tailles manquantes: {', '.join(missing_sizes)}")

    # 2) Fichiers requis par réplication + 3) lignes non nulles par taille
    rows_by_size: dict[int, int] = {size: 0 for size in EXPECTED_SIZES}

    for size, size_dir in size_dirs.items():
        rep_dirs = _iter_rep_dirs(size_dir)
        if not rep_dirs:
            failures.append(f"{size_dir.name}: aucun dossier rep_* trouvé")
            continue

        for rep_dir in rep_dirs:
            raw_csvs = sorted(rep_dir.glob("raw_*.csv"))
            agg_csv = rep_dir / "aggregated_results.csv"

            if not raw_csvs:
                failures.append(f"{rep_dir.relative_to(results_dir)}: raw_*.csv absent")
            if not agg_csv.exists():
                failures.append(
                    f"{rep_dir.relative_to(results_dir)}: aggregated_results.csv absent"
                )

            for csv_path in [*raw_csvs, agg_csv]:
                if not csv_path.exists():
                    continue
                try:
                    rows_by_size[size] += _count_data_rows(csv_path)
                except Exception as exc:
                    failures.append(
                        f"{csv_path.relative_to(results_dir)}: lecture impossible ({exc})"
                    )

    print("\nRésumé lignes de données par taille:")
    for size in EXPECTED_SIZES:
        print(f"- size_{size}: {rows_by_size[size]} ligne(s)")
        if size in size_dirs and rows_by_size[size] == 0:
            failures.append(f"size_{size}: 0 ligne de données cumulée")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
    else:
        print("\nPASS")

    # Exigence explicite: code non-zéro si une taille manque.
    if missing_sizes:
        return 2

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
