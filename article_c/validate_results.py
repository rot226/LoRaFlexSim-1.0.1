"""Valide les résultats générés pour l'article C."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


class AnomalyTracker:
    def __init__(self, max_samples: int = 20) -> None:
        self.count = 0
        self.max_samples = max_samples
        self.samples: list[str] = []

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.samples) < self.max_samples:
            self.samples.append(message)

    def has_anomalies(self) -> bool:
        return self.count > 0


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _read_csv(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, object]] = [row for row in reader]
    return rows, fieldnames


def _check_constant(
    values: list[float],
    label: str,
    tracker: AnomalyTracker,
    const_tolerance: float,
) -> None:
    if len(values) < 2:
        tracker.add(f"{label}: valeurs insuffisantes pour vérifier la variation.")
        return
    if max(values) - min(values) <= const_tolerance:
        tracker.add(f"{label}: valeur quasi constante détectée.")


def _check_range(
    values: list[float],
    label: str,
    tracker: AnomalyTracker,
) -> None:
    for idx, value in enumerate(values, start=1):
        if value < 0.0 or value > 1.0:
            tracker.add(
                f"{label}: valeur hors [0,1] à la ligne {idx}: {value:.6f}."
            )


def _check_received_formula(
    rows: list[dict[str, object]],
    sent_key: str,
    received_key: str,
    pdr_key: str,
    tolerance: float,
    label: str,
    tracker: AnomalyTracker,
) -> None:
    for idx, row in enumerate(rows, start=1):
        sent = _parse_float(row.get(sent_key))
        received = _parse_float(row.get(received_key))
        pdr = _parse_float(row.get(pdr_key))
        if sent is None or received is None or pdr is None:
            tracker.add(
                f"{label}: valeurs manquantes à la ligne {idx} pour {sent_key}, "
                f"{received_key} ou {pdr_key}."
            )
            continue
        expected = sent * pdr
        diff = abs(received - expected)
        limit = max(1.0, abs(expected)) * tolerance
        if diff > limit:
            tracker.add(
                f"{label}: incohérence ligne {idx} (received={received:.6f}, "
                f"sent*pdr={expected:.6f}, diff={diff:.6f})."
            )


def _validate_pdr_file(
    path: Path,
    pdr_key: str,
    sent_key: str,
    received_key: str,
    tolerance: float,
    const_tolerance: float,
    tracker: AnomalyTracker,
) -> None:
    rows, fieldnames = _read_csv(path)
    label = f"{path}"
    if not fieldnames:
        tracker.add(f"{label}: fichier manquant ou vide.")
        return
    missing_columns = [key for key in (pdr_key, sent_key, received_key) if key not in fieldnames]
    if missing_columns:
        tracker.add(
            f"{label}: colonnes manquantes {', '.join(missing_columns)}."
        )
        return
    pdr_values = [
        value
        for value in (_parse_float(row.get(pdr_key)) for row in rows)
        if value is not None
    ]
    if not pdr_values:
        tracker.add(f"{label}: aucune valeur de PDR exploitable.")
        return
    _check_range(pdr_values, f"{label} ({pdr_key})", tracker)
    _check_constant(
        pdr_values,
        f"{label} ({pdr_key})",
        tracker,
        const_tolerance,
    )
    _check_received_formula(
        rows,
        sent_key,
        received_key,
        pdr_key,
        tolerance,
        label,
        tracker,
    )


def _validate_reward_file(
    path: Path,
    reward_key: str,
    const_tolerance: float,
    tracker: AnomalyTracker,
) -> None:
    rows, fieldnames = _read_csv(path)
    label = f"{path}"
    if not fieldnames:
        tracker.add(f"{label}: fichier manquant ou vide.")
        return
    if reward_key not in fieldnames:
        tracker.add(f"{label}: colonne {reward_key} absente.")
        return
    reward_values = [
        value
        for value in (_parse_float(row.get(reward_key)) for row in rows)
        if value is not None
    ]
    if not reward_values:
        tracker.add(f"{label}: aucune valeur reward exploitable.")
        return
    _check_constant(
        reward_values,
        f"{label} ({reward_key})",
        tracker,
        const_tolerance,
    )


def validate_results(
    step1_dir: Path,
    step2_dir: Path,
    tolerance: float,
    const_tolerance: float,
    max_samples: int,
) -> AnomalyTracker:
    tracker = AnomalyTracker(max_samples=max_samples)

    _validate_pdr_file(
        step1_dir / "raw_metrics.csv",
        pdr_key="pdr",
        sent_key="sent",
        received_key="received",
        tolerance=tolerance,
        const_tolerance=const_tolerance,
        tracker=tracker,
    )
    _validate_pdr_file(
        step1_dir / "aggregated_results.csv",
        pdr_key="pdr_mean",
        sent_key="sent_mean",
        received_key="received_mean",
        tolerance=tolerance,
        const_tolerance=const_tolerance,
        tracker=tracker,
    )
    _validate_reward_file(
        step2_dir / "raw_results.csv",
        reward_key="reward",
        const_tolerance=const_tolerance,
        tracker=tracker,
    )
    _validate_reward_file(
        step2_dir / "aggregated_results.csv",
        reward_key="reward_mean",
        const_tolerance=const_tolerance,
        tracker=tracker,
    )
    return tracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide PDR, cohérence received/sent et variation des rewards."
    )
    parser.add_argument(
        "--step1-dir",
        type=Path,
        default=Path("article_c/step1/results"),
        help="Répertoire des résultats de l'étape 1.",
    )
    parser.add_argument(
        "--step2-dir",
        type=Path,
        default=Path("article_c/step2/results"),
        help="Répertoire des résultats de l'étape 2.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolérance relative pour received ≈ sent*pdr.",
    )
    parser.add_argument(
        "--const-tolerance",
        type=float,
        default=1e-4,
        help="Tolérance pour juger une valeur quasi constante.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Nombre maximal d'anomalies détaillées à afficher.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tracker = validate_results(
        step1_dir=args.step1_dir,
        step2_dir=args.step2_dir,
        tolerance=args.tolerance,
        const_tolerance=args.const_tolerance,
        max_samples=args.max_samples,
    )
    if tracker.has_anomalies():
        print(f"Anomalies détectées: {tracker.count}.")
        for message in tracker.samples:
            print(f"- {message}")
        if tracker.count > len(tracker.samples):
            remaining = tracker.count - len(tracker.samples)
            print(f"- ... et {remaining} anomalies supplémentaires.")
        return 1
    print("Aucune anomalie détectée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
