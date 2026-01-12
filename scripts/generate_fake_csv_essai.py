"""Génère des CSV fictifs (synthetic=true) pour l'essai Step1/Step2.

Les distributions sont plausibles :
- SNIR on < SNIR off
- DER < 1

Exemple :
    python scripts/generate_fake_csv_essai.py --output-dir results/fake_essai
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

DEFAULT_ALGOS = ("adr", "apra", "mixra_h", "mixra_opt")


def _truncated_normal(
    rng: random.Random,
    mean: float,
    std: float,
    min_value: float,
    max_value: float,
) -> float:
    for _ in range(100):
        value = rng.gauss(mean, std)
        if min_value <= value <= max_value:
            return value
    return max(min_value, min(max_value, mean))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "fake_essai",
        help="Répertoire de sortie pour les CSV fictifs",
    )
    parser.add_argument(
        "--rows-per-config",
        type=int,
        default=12,
        help="Nombre de lignes par couple (algo, SNIR)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour stabiliser les résultats",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(DEFAULT_ALGOS),
        help="Liste des algorithmes à inclure dans les CSV",
    )
    return parser


def _generate_step1_rows(
    rng: random.Random,
    algos: list[str],
    rows_per_config: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for algo in algos:
        for use_snir in (False, True):
            if use_snir:
                snir_mean = (4.5, 1.2, 0.5, 9.0)
                der_range = (0.45, 0.85)
                pdr_range = (0.6, 0.92)
                collisions_range = (40, 140)
            else:
                snir_mean = (8.5, 1.5, 4.0, 14.0)
                der_range = (0.55, 0.92)
                pdr_range = (0.7, 0.98)
                collisions_range = (20, 100)

            for _ in range(rows_per_config):
                der = _truncated_normal(rng, mean=sum(der_range) / 2, std=0.08, min_value=0.2, max_value=0.98)
                pdr = _truncated_normal(rng, mean=sum(pdr_range) / 2, std=0.07, min_value=0.3, max_value=0.99)
                snir_value = _truncated_normal(rng, *snir_mean)
                collisions = rng.randint(*collisions_range)

                rows.append(
                    {
                        "PDR": f"{pdr:.3f}",
                        "DER": f"{der:.3f}",
                        "collisions": str(collisions),
                        "snir_mean": f"{snir_value:.3f}",
                        "use_snir": "true" if use_snir else "false",
                        "algo": algo,
                        "synthetic": "true",
                    }
                )
    return rows


def _generate_step2_rows(
    rng: random.Random,
    rows: int,
) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    for _ in range(rows):
        der = _truncated_normal(rng, mean=0.7, std=0.08, min_value=0.3, max_value=0.98)
        pdr = _truncated_normal(rng, mean=0.82, std=0.06, min_value=0.4, max_value=0.99)
        snir_mean = _truncated_normal(rng, mean=6.2, std=1.6, min_value=2.0, max_value=12.0)
        reward = max(0.0, min(1.0, 0.6 * pdr + 0.4 * der - rng.uniform(0.0, 0.15)))
        regret = max(0.0, min(0.6, 1.0 - reward + rng.uniform(0.0, 0.1)))

        rows_out.append(
            {
                "reward": f"{reward:.3f}",
                "regret": f"{regret:.3f}",
                "der": f"{der:.3f}",
                "pdr": f"{pdr:.3f}",
                "snir_mean": f"{snir_mean:.3f}",
                "synthetic": "true",
            }
        )
    return rows_out


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    step1_rows = _generate_step1_rows(rng, list(args.algos), args.rows_per_config)
    step2_rows = _generate_step2_rows(rng, len(step1_rows))

    step1_path = args.output_dir / "step1_fake.csv"
    step2_path = args.output_dir / "step2_fake.csv"

    _write_csv(step1_path, step1_rows)
    _write_csv(step2_path, step2_rows)

    print(f"[OK] CSV Step1 fictif: {step1_path}")
    print(f"[OK] CSV Step2 fictif: {step2_path}")


if __name__ == "__main__":
    main()
