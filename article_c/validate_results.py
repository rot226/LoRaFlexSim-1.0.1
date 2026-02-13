"""Valide les résultats générés pour l'article C."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from importlib.util import find_spec
from pathlib import Path

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


class AnomalyTracker:
    def __init__(self, max_samples: int = 20) -> None:
        self.count = 0
        self.max_samples = max_samples
        self.samples: list[str] = []
        self.packet_level_rows = 0

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.samples) < self.max_samples:
            self.samples.append(message)

    def has_anomalies(self) -> bool:
        return self.count > 0

    def add_packet_level_rows(self, count: int) -> None:
        if count > 0:
            self.packet_level_rows += count


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


def _collect_size_aggregated_csvs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("by_size/size_*/rep_*/aggregated_results.csv"))


def _check_by_size_coverage(results_dir: Path, tracker: AnomalyTracker, step_label: str) -> None:
    by_size_paths = _collect_size_aggregated_csvs(results_dir)
    if not by_size_paths:
        tracker.add(f"{step_label}: aucun agrégat by_size trouvé sous {results_dir / 'by_size'}.")


def _check_required_replication_files(
    results_dir: Path,
    tracker: AnomalyTracker,
    step_label: str,
    required_files: tuple[str, ...],
) -> None:
    rep_dirs = sorted(
        path
        for path in results_dir.glob("by_size/size_*/rep_*")
        if path.is_dir()
    )
    if not rep_dirs:
        tracker.add(
            f"{step_label}: aucun dossier de réplication trouvé sous {results_dir / 'by_size'} (attendu: by_size/size_<N>/rep_<R>)."
        )
        return
    for rep_dir in rep_dirs:
        for filename in required_files:
            expected = rep_dir / filename
            if not expected.exists():
                tracker.add(f"{step_label}: fichier obligatoire manquant: {expected}.")


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
    packet_level_rows = 0
    for idx, row in enumerate(rows, start=1):
        if not {sent_key, received_key, pdr_key}.issubset(row.keys()):
            packet_level_rows += 1
            continue
        sent = _parse_float(row.get(sent_key))
        received = _parse_float(row.get(received_key))
        pdr = _parse_float(row.get(pdr_key))
        if sent is None or received is None or pdr is None:
            packet_level_rows += 1
            continue
        expected = sent * pdr
        diff = abs(received - expected)
        limit = max(1.0, abs(expected)) * tolerance
        if diff > limit:
            tracker.add(
                f"{label}: incohérence ligne {idx} (received={received:.6f}, "
                f"sent*pdr={expected:.6f}, diff={diff:.6f})."
            )
    tracker.add_packet_level_rows(packet_level_rows)


def _use_received_algo_mean(
    rows: list[dict[str, object]],
    sent_key: str,
    pdr_key: str,
    received_key: str,
    tolerance: float,
    label: str,
    tracker: AnomalyTracker,
) -> str:
    if sent_key != "sent_mean" or pdr_key != "pdr_mean" or received_key != "received_mean":
        return received_key
    derived_key = "received_algo_mean"
    for idx, row in enumerate(rows, start=1):
        sent = _parse_float(row.get(sent_key))
        pdr = _parse_float(row.get(pdr_key))
        if sent is None or pdr is None:
            continue
        derived_value = sent * pdr
        row[derived_key] = derived_value
        received = _parse_float(row.get(received_key))
        if received is None:
            continue
        diff = abs(received - derived_value)
        limit = max(1.0, abs(derived_value)) * tolerance
        if diff > limit:
            tracker.add(
                f"{label}: received_mean diverge de sent_mean*pdr_mean "
                f"ligne {idx} (received={received:.6f}, "
                f"derived={derived_value:.6f}, diff={diff:.6f})."
            )
    return derived_key


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
    received_key = _use_received_algo_mean(
        rows,
        sent_key,
        pdr_key,
        received_key,
        tolerance,
        label,
        tracker,
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
    skip_step2: bool,
) -> AnomalyTracker:
    tracker = AnomalyTracker(max_samples=max_samples)
    _check_by_size_coverage(step1_dir, tracker, "Step1")
    _check_required_replication_files(
        step1_dir,
        tracker,
        "Step1",
        required_files=("raw_packets.csv", "raw_metrics.csv", "aggregated_results.csv"),
    )
    if not skip_step2:
        _check_by_size_coverage(step2_dir, tracker, "Step2")

    _validate_pdr_file(
        step1_dir / "aggregates" / "aggregated_results.csv",
        pdr_key="pdr_mean",
        sent_key="sent_mean",
        received_key="received_mean",
        tolerance=tolerance,
        const_tolerance=const_tolerance,
        tracker=tracker,
    )
    if not skip_step2:
        _validate_reward_file(
            step2_dir / "aggregates" / "aggregated_results.csv",
            reward_key="reward_mean",
            const_tolerance=const_tolerance,
            tracker=tracker,
        )
    return tracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide les agrégats globaux et par taille de l'article C."
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
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Ignore les contrôles de l'étape 2.",
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
        skip_step2=args.skip_step2,
    )
    scope_label = "Step1" if args.skip_step2 else "résultats"
    if tracker.has_anomalies():
        print(f"Anomalies détectées ({scope_label}): {tracker.count}.")
        for message in tracker.samples:
            print(f"- {message}")
        if tracker.count > len(tracker.samples):
            remaining = tracker.count - len(tracker.samples)
            print(f"- ... et {remaining} anomalies supplémentaires.")
        return 1
    if tracker.packet_level_rows:
        print(
            "Validation info: "
            f"{tracker.packet_level_rows} ligne(s) packet-level ignorée(s)."
        )
    print(f"Aucune anomalie {scope_label} détectée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
