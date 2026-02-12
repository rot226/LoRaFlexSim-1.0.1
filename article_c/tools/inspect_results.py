"""Inspection rapide des résultats CSV pour Step1/Step2."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from article_c.common.config import BASE_DIR

STEP_TARGETS: dict[str, tuple[str, ...]] = {
    "step1": (
        "aggregated_results.csv",
        "aggregated_results_by_replication.csv",
        "aggregated_results_by_round.csv",
        "raw_metrics.csv",
        "raw_packets.csv",
    ),
    "step2": (
        "aggregated_results.csv",
        "aggregated_results_by_replication.csv",
        "aggregated_results_by_round.csv",
        "raw_results.csv",
        "raw_all.csv",
        "raw_cluster.csv",
    ),
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "step1": ("network_size", "algo", "snir_mode", "cluster"),
    "step2": ("network_size", "algo", "snir_mode", "cluster"),
}

NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "step1": ("network_size", "sent", "received", "pdr"),
    "step2": ("network_size", "reward", "success_rate"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspecte les résultats CSV de Step1/Step2."
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=("step1", "step2"),
        help="Étape à inspecter.",
    )
    return parser


def _detect_target_csv(step: str) -> tuple[Path, list[Path]]:
    results_dir = BASE_DIR / step / "results"
    targets = [results_dir / name for name in STEP_TARGETS[step]]
    existing = [path for path in targets if path.exists()]
    return results_dir, existing


def _fmt_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.2f} MiB"


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _safe_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _inspect_schema_and_types(
    step: str,
    path: Path,
    header: list[str],
    rows: Iterable[dict[str, str]],
) -> list[str]:
    warnings: list[str] = []
    required = REQUIRED_COLUMNS[step]
    missing = [column for column in required if column not in header]
    if missing:
        warnings.append(
            f"{path.name}: colonnes manquantes -> {', '.join(missing)}"
        )

    numeric_columns = [c for c in NUMERIC_COLUMNS[step] if c in header]
    for idx, row in enumerate(rows, start=2):
        for column in numeric_columns:
            raw_value = row.get(column, "")
            if str(raw_value).strip() == "":
                continue
            number = _safe_float(raw_value)
            if number is None:
                warnings.append(
                    f"{path.name}: type invalide en ligne {idx} pour {column}='{raw_value}'"
                )
                continue
            if column == "network_size" and (number <= 0 or not number.is_integer()):
                warnings.append(
                    f"{path.name}: network_size invalide ligne {idx} ({raw_value})"
                )
            if column in {"pdr", "success_rate"} and not (0.0 <= number <= 1.0):
                warnings.append(
                    f"{path.name}: {column} hors [0,1] ligne {idx} ({raw_value})"
                )
    return warnings


def _count_by_keys(rows: Iterable[dict[str, str]]) -> Counter[tuple[str, str, str, str]]:
    counter: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        key = (
            str(row.get("network_size", "<NA>") or "<NA>").strip(),
            str(row.get("algo", "<NA>") or "<NA>").strip(),
            str(row.get("snir_mode", "<NA>") or "<NA>").strip(),
            str(row.get("cluster", "<NA>") or "<NA>").strip(),
        )
        counter[key] += 1
    return counter


def main() -> int:
    args = _build_parser().parse_args()
    step = args.step

    results_dir, csv_paths = _detect_target_csv(step)
    print(f"Dossier résultats: {results_dir}")
    if not csv_paths:
        print("Aucun CSV cible détecté.")
        return 1

    print("CSV cibles détectés:")
    for path in csv_paths:
        print(f"- {path.name}: {_fmt_size(path)}")

    global_counter: Counter[tuple[str, str, str, str]] = Counter()
    all_warnings: list[str] = []
    detected_sizes: set[int] = set()

    for path in csv_paths:
        header, rows = _read_rows(path)
        file_counter = _count_by_keys(rows)
        global_counter.update(file_counter)

        if "network_size" in header:
            for row in rows:
                maybe_size = _safe_float(row.get("network_size", ""))
                if maybe_size is not None and maybe_size.is_integer() and maybe_size > 0:
                    detected_sizes.add(int(maybe_size))

        all_warnings.extend(_inspect_schema_and_types(step, path, header, rows))

    sizes_label = ", ".join(map(str, sorted(detected_sizes))) if detected_sizes else "<aucune>"
    print(f"Tailles détectées: {sizes_label}")

    print("\nComptage lignes par (taille, algo, snir_mode, cluster):")
    for key, count in sorted(global_counter.items()):
        print(f"- {key}: {count}")

    if step == "step1" and len(detected_sizes) < 2:
        print("ATTENTION: moins de 2 tailles détectées pour Step1.")

    if all_warnings:
        print("\nAvertissements schéma/type:")
        seen = set()
        for warning in all_warnings:
            if warning in seen:
                continue
            seen.add(warning)
            print(f"- {warning}")
    else:
        print("\nAucun avertissement schéma/type.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
