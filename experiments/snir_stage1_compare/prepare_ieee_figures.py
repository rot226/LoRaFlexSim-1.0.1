"""Préparation des figures IEEE à partir des CSV `_compare`.

Ce script agrège les fichiers générés par `run_compare_stage1.py` en
calculant la moyenne et l'intervalle de confiance à 95 % des principales
métriques par combinaison (profil PHY, nombre de noeuds, intervalle). Les
résultats sont enregistrés dans de nouveaux fichiers suffixés `_ieee.csv`
dans le même répertoire que les données d'entrée.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIDENCE_Z = 1.96

# Colonnes numériques attendues dans les CSV `_compare`
NUMERIC_FIELDS: Mapping[str, type] = {
    "num_nodes": int,
    "packet_interval_s": float,
    "rep": int,
    "seed": int,
    "packets_sent": int,
    "packets_attempted": int,
    "packets_delivered": int,
    "collisions": int,
    "der": float,
    "pdr": float,
    "throughput_bps": float,
    "snir_mean": float,
    "snir_median": float,
    "sim_time_s": float,
}

# Métriques à exporter dans les fichiers `_ieee.csv` avec moyenne et IC95 %
METRICS = (
    "der",
    "pdr",
    "throughput_bps",
    "snir_mean",
    "snir_median",
    "collisions",
    "packets_sent",
    "packets_attempted",
    "packets_delivered",
    "sim_time_s",
)


def _parse_numeric(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = dict(row)
    for field, caster in NUMERIC_FIELDS.items():
        if field in parsed and parsed[field] != "":
            parsed[field] = caster(parsed[field])
    return parsed


def _mean_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0, 0.0

    avg = statistics.mean(values)
    if len(values) == 1:
        return avg, avg, avg

    std_dev = statistics.stdev(values)
    margin = CONFIDENCE_Z * std_dev / math.sqrt(len(values))
    return avg, avg - margin, avg + margin


def _aggregate_group(rows: list[dict[str, object]]) -> dict[str, object]:
    sample_count = len(rows)
    base = {
        "algorithm": rows[0]["algorithm"],
        "phy_profile": rows[0]["phy_profile"],
        "num_nodes": rows[0]["num_nodes"],
        "packet_interval_s": rows[0]["packet_interval_s"],
        "samples": sample_count,
    }

    for metric in METRICS:
        mean_val, ci_low, ci_high = _mean_ci95(
            row[metric] for row in rows if metric in row
        )
        base[f"{metric}_mean"] = mean_val
        base[f"{metric}_ci95_low"] = ci_low
        base[f"{metric}_ci95_high"] = ci_high

    return base


def _collect_groups(path: Path) -> dict[tuple, list[dict[str, object]]]:
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = _parse_numeric(row)
            key = (
                parsed.get("algorithm"),
                parsed.get("phy_profile"),
                parsed.get("num_nodes"),
                parsed.get("packet_interval_s"),
            )
            groups[key].append(parsed)
    return groups


def _write_ieee_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    metric_fields: list[str] = []
    for metric in METRICS:
        metric_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ]
        )

    fieldnames = [
        "algorithm",
        "phy_profile",
        "num_nodes",
        "packet_interval_s",
        "samples",
        *metric_fields,
    ]

    with path.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_compare_csv(path: Path) -> Path:
    """Transforme un CSV `_compare` en CSV `_ieee` avec moyenne et IC95 %."""

    groups = _collect_groups(path)
    aggregated = [_aggregate_group(rows) for rows in groups.values()]
    aggregated.sort(key=lambda r: (r["phy_profile"], r["num_nodes"], r["packet_interval_s"]))

    target = path.with_name(path.name.replace("_compare", "_ieee"))
    _write_ieee_csv(target, aggregated)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule la moyenne et l'IC95 % des métriques à partir des fichiers"
            " _compare et génère les équivalents _ieee.csv"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire contenant les CSV _compare (défaut : data/).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Répertoire {data_dir} introuvable")
        return 1

    compare_files = sorted(data_dir.glob("*_compare.csv"))
    if not compare_files:
        print(f"Aucun fichier _compare trouvé dans {data_dir}")
        return 1

    written: list[Path] = []
    for csv_path in compare_files:
        target = process_compare_csv(csv_path)
        written.append(target)

    print("Fichiers générés :")
    for path in written:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
