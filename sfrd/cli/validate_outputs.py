"""Point d'entrée CLI: validation des sorties."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
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
_EXPECTED_NETWORK_SIZES = {80, 160, 320, 640, 1280}
_EXPECTED_ALGORITHMS = {"UCB", "ADR", "MixRA-H", "MixRA-Opt"}
_EXPECTED_SNIR = {"OFF", "ON"}


@dataclass(frozen=True)
class ValidationAnomaly:
    severity: str
    category: str
    message: str


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


def _collect_unique_runs(output_root: Path) -> set[tuple[int, str, str]]:
    runs: set[tuple[int, str, str]] = set()
    for snir in _EXPECTED_SNIR:
        csv_path = output_root / f"SNIR_{snir}" / "pdr_results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    size = int((row.get("network_size") or "").strip())
                except ValueError:
                    continue
                algorithm = (row.get("algorithm") or "").strip()
                snir_value = (row.get("snir") or "").strip().upper()
                if algorithm and snir_value:
                    runs.add((size, algorithm, snir_value))
    return runs


def _validate_matrix_completeness(output_root: Path) -> list[ValidationAnomaly]:
    anomalies: list[ValidationAnomaly] = []
    expected = {
        (size, algo, snir)
        for size in _EXPECTED_NETWORK_SIZES
        for algo in _EXPECTED_ALGORITHMS
        for snir in _EXPECTED_SNIR
    }
    realized = _collect_unique_runs(output_root)
    missing = sorted(expected - realized)
    if missing:
        preview = ", ".join(f"{size}/{algo}/{snir}" for size, algo, snir in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        anomalies.append(
            ValidationAnomaly(
                severity="critical",
                category="matrix_completeness",
                message=(
                    "Matrice incomplète (5 tailles x 4 algos x 2 SNIR): "
                    f"{len(missing)} combinaison(s) manquante(s). Exemple(s): {preview}{suffix}"
                ),
            )
        )
    return anomalies


def _validate_internal_coherence(output_root: Path) -> list[ValidationAnomaly]:
    anomalies: list[ValidationAnomaly] = []
    for csv_path in sorted(output_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            if not {"success", "tx", "pdr"}.issubset(fieldnames):
                continue
            for row_number, row in enumerate(reader, start=2):
                try:
                    success = _parse_float(row["success"], "success", csv_path, row_number)
                    tx = _parse_float(row["tx"], "tx", csv_path, row_number)
                    pdr = _parse_float(row["pdr"], "pdr", csv_path, row_number)
                except ValueError as exc:
                    anomalies.append(
                        ValidationAnomaly(
                            severity="critical",
                            category="internal_consistency",
                            message=str(exc),
                        )
                    )
                    continue
                if success < 0.0 or tx < 0.0:
                    anomalies.append(
                        ValidationAnomaly(
                            severity="critical",
                            category="internal_consistency",
                            message=(
                                f"[{csv_path}] ligne {row_number}: success/tx doivent être >= 0 "
                                f"(success={success}, tx={tx})"
                            ),
                        )
                    )
                    continue
                if success > tx:
                    anomalies.append(
                        ValidationAnomaly(
                            severity="critical",
                            category="internal_consistency",
                            message=(
                                f"[{csv_path}] ligne {row_number}: success ({success}) > tx ({tx})"
                            ),
                        )
                    )
                expected_pdr = 0.0 if tx == 0 else success / tx
                if abs(pdr - expected_pdr) > 1e-6:
                    anomalies.append(
                        ValidationAnomaly(
                            severity="critical",
                            category="internal_consistency",
                            message=(
                                f"[{csv_path}] ligne {row_number}: incohérence pdr (pdr={pdr}, "
                                f"success/tx={expected_pdr})"
                            ),
                        )
                    )
    return anomalies


def _campaign_date(output_root: Path) -> str:
    required_paths = [output_root / relative_path for relative_path, _ in _REQUIRED_CSVS]
    existing = [path for path in required_paths if path.exists()]
    if not existing:
        return datetime.now().isoformat(timespec="seconds")
    newest = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(newest).isoformat(timespec="seconds")


def _write_release_report(
    output_root: Path,
    *,
    validated_files: list[Path],
    anomalies: list[ValidationAnomaly],
    realized_runs: int,
) -> Path:
    report_path = output_root / "release_report.txt"
    expected_runs = len(_EXPECTED_NETWORK_SIZES) * len(_EXPECTED_ALGORITHMS) * len(_EXPECTED_SNIR)
    lines: list[str] = [
        "SFRD Release Report",
        "===================",
        f"Date campagne: {_campaign_date(output_root)}",
        f"Runs attendus: {expected_runs}",
        f"Runs réalisés: {realized_runs}",
        "",
        "CSV validés:",
    ]
    if validated_files:
        lines.extend(f"- {path.as_posix()}" for path in validated_files)
    else:
        lines.append("- Aucun")

    lines.extend(["", "Anomalies détectées:"])
    if anomalies:
        for anomaly in anomalies:
            lines.append(f"- [{anomaly.severity}] ({anomaly.category}) {anomaly.message}")
    else:
        lines.append("- Aucune")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    output_root: Path = args.output_root

    anomalies: list[ValidationAnomaly] = []
    validated_files: list[Path] = []
    for relative_path, expected_columns in _REQUIRED_CSVS:
        csv_path = output_root / relative_path
        try:
            _validate_csv(csv_path, expected_columns)
            validated_files.append(csv_path)
            print(f"[OK] {csv_path}")
        except (FileNotFoundError, ValueError) as exc:
            anomalies.append(
                ValidationAnomaly(
                    severity="critical",
                    category="csv_validation",
                    message=str(exc),
                )
            )
            print(f"[ERROR] {exc}")

    anomalies.extend(_validate_internal_coherence(output_root))
    anomalies.extend(_validate_matrix_completeness(output_root))
    realized_runs = len(_collect_unique_runs(output_root))
    report_path = _write_release_report(
        output_root,
        validated_files=validated_files,
        anomalies=anomalies,
        realized_runs=realized_runs,
    )
    has_critical = any(anomaly.severity == "critical" for anomaly in anomalies)

    print(f"Rapport release: {report_path}")
    if has_critical:
        print(f"Validation release échouée: {len(anomalies)} anomalie(s), dont critique(s).")
        sys.exit(1)

    print("Validation release réussie: aucune anomalie critique.")


if __name__ == "__main__":
    main()
