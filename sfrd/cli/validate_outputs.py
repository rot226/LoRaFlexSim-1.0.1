"""Point d'entrée CLI: validation des sorties."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

_REQUIRED_CSVS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "SNIR_OFF/pdr_results.csv",
        ("network_size", "algorithm", "snir", "pdr"),
    ),
    (
        "SNIR_OFF/throughput_results.csv",
        ("network_size", "algorithm", "snir", "throughput_packets_per_s"),
    ),
    (
        "SNIR_OFF/energy_results.csv",
        ("network_size", "algorithm", "snir", "energy_joule_per_packet"),
    ),
    (
        "SNIR_OFF/sf_distribution.csv",
        ("network_size", "algorithm", "snir", "sf", "count"),
    ),
    (
        "SNIR_ON/pdr_results.csv",
        ("network_size", "algorithm", "snir", "pdr"),
    ),
    (
        "SNIR_ON/throughput_results.csv",
        ("network_size", "algorithm", "snir", "throughput_packets_per_s"),
    ),
    (
        "SNIR_ON/energy_results.csv",
        ("network_size", "algorithm", "snir", "energy_joule_per_packet"),
    ),
    (
        "SNIR_ON/sf_distribution.csv",
        ("network_size", "algorithm", "snir", "sf", "count"),
    ),
    (
        "learning_curve_ucb.csv",
        ("episode", "reward"),
    ),
)

_ALLOWED_SF = {7, 8, 9, 10, 11, 12}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valide les CSV de sortie SFRD.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("sfrd/output"),
        help="Dossier racine contenant SNIR_OFF/, SNIR_ON/ et learning_curve_ucb.csv",
    )
    return parser.parse_args()


def _is_nan_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.lower() == "nan":
        return True
    try:
        parsed = float(text)
    except ValueError:
        return False
    return math.isnan(parsed)


def _parse_float(value: str, field_name: str, csv_path: Path, row_number: int) -> float:
    text = value.strip()
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"[{csv_path}] ligne {row_number}: valeur numérique invalide pour '{field_name}': {value!r}"
        ) from exc

    if math.isnan(parsed):
        raise ValueError(
            f"[{csv_path}] ligne {row_number}: NaN interdit pour '{field_name}'"
        )
    return parsed


def _parse_int(value: str, field_name: str, csv_path: Path, row_number: int) -> int:
    text = value.strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"[{csv_path}] ligne {row_number}: entier invalide pour '{field_name}': {value!r}"
        ) from exc


def _validate_columns(csv_path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise ValueError(
                f"[{csv_path}] colonnes invalides. Attendu: {list(expected_columns)} ; obtenu: {list(actual_columns)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"[{csv_path}] fichier CSV vide (aucune ligne de données)")
    return rows


def _validate_no_nan(rows: Iterable[dict[str, str]], csv_path: Path) -> None:
    for row_number, row in enumerate(rows, start=2):
        for field_name, raw_value in row.items():
            if _is_nan_text(raw_value):
                raise ValueError(
                    f"[{csv_path}] ligne {row_number}: NaN interdit dans la colonne '{field_name}'"
                )


def _validate_snir_folder(rows: Iterable[dict[str, str]], csv_path: Path, expected: str) -> None:
    for row_number, row in enumerate(rows, start=2):
        snir = row["snir"].strip().upper()
        if snir != expected:
            raise ValueError(
                f"[{csv_path}] ligne {row_number}: snir incohérent (attendu {expected}, obtenu {row['snir']!r})"
            )


def _validate_business_rules(rows: Iterable[dict[str, str]], csv_path: Path) -> None:
    metric_name = csv_path.stem

    for row_number, row in enumerate(rows, start=2):
        if "pdr" in row:
            pdr = _parse_float(row["pdr"], "pdr", csv_path, row_number)
            if not (0.0 <= pdr <= 1.0):
                raise ValueError(
                    f"[{csv_path}] ligne {row_number}: contrainte violée 0 <= pdr <= 1 (valeur={pdr})"
                )

        if "throughput_packets_per_s" in row:
            throughput = _parse_float(
                row["throughput_packets_per_s"],
                "throughput_packets_per_s",
                csv_path,
                row_number,
            )
            if throughput < 0.0:
                raise ValueError(
                    f"[{csv_path}] ligne {row_number}: contrainte violée throughput_packets_per_s >= 0 (valeur={throughput})"
                )

        if "energy_joule_per_packet" in row:
            energy = _parse_float(
                row["energy_joule_per_packet"],
                "energy_joule_per_packet",
                csv_path,
                row_number,
            )
            if energy < 0.0:
                raise ValueError(
                    f"[{csv_path}] ligne {row_number}: contrainte violée energy_joule_per_packet >= 0 (valeur={energy})"
                )

        if metric_name == "sf_distribution":
            sf = _parse_int(row["sf"], "sf", csv_path, row_number)
            if sf not in _ALLOWED_SF:
                raise ValueError(
                    f"[{csv_path}] ligne {row_number}: sf invalide {sf}. Valeurs autorisées: {sorted(_ALLOWED_SF)}"
                )


def _validate_csv(csv_path: Path, expected_columns: tuple[str, ...]) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV requis manquant: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Fichier CSV requis vide (taille 0): {csv_path}")

    rows = _validate_columns(csv_path, expected_columns)
    _validate_no_nan(rows, csv_path)

    if "snir" in expected_columns:
        expected_snir = "ON" if "SNIR_ON" in csv_path.parts else "OFF"
        _validate_snir_folder(rows, csv_path, expected_snir)

    _validate_business_rules(rows, csv_path)


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    output_root: Path = args.output_root

    errors: list[str] = []
    for relative_path, expected_columns in _REQUIRED_CSVS:
        csv_path = output_root / relative_path
        try:
            _validate_csv(csv_path, expected_columns)
            print(f"[OK] {csv_path}")
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            print(f"[ERROR] {exc}")

    if errors:
        print(f"Validation échouée: {len(errors)} erreur(s).")
        sys.exit(1)

    print("Validation réussie: tous les CSV requis sont conformes.")


if __name__ == "__main__":
    main()
