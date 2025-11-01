"""Compare les séries normalisées du banc QoS aux valeurs de référence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.scenarios.qos_cluster_bench import VALIDATION_MODE

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "qos_clusters" / VALIDATION_MODE
DEFAULT_SERIES_PATH = DEFAULT_RESULTS_DIR / "validation_normalized_metrics.json"
DEFAULT_REFERENCE_PATH = ROOT_DIR / "docs" / "qos_validation_reference.json"


@dataclass(frozen=True)
class EntryKey:
    """Identifie un point de série par algorithme et charge."""

    algorithm: str
    num_nodes: int
    packet_interval_s: float

    def as_tuple(self) -> Tuple[str, int, float]:
        return (self.algorithm, self.num_nodes, self.packet_interval_s)


def _load_payload(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Format inattendu pour {path}: dictionnaire attendu")
    return data


def _index_entries(payload: Mapping[str, object]) -> Dict[EntryKey, Mapping[str, object]]:
    entries = payload.get("entries", [])
    if not isinstance(entries, Iterable):
        raise ValueError("Champ 'entries' absent ou invalide dans les séries normalisées")
    indexed: Dict[EntryKey, Mapping[str, object]] = {}
    for raw in entries:
        if not isinstance(raw, Mapping):
            continue
        algorithm = str(raw.get("algorithm", ""))
        num_nodes = int(raw.get("num_nodes", 0) or 0)
        period = float(raw.get("packet_interval_s", 0.0) or 0.0)
        key = EntryKey(algorithm, num_nodes, period)
        indexed[key] = raw
    return indexed


def _compare_numeric(
    actual: Mapping[str, object],
    reference: Mapping[str, object],
    fields: Iterable[str],
    tolerance: float,
    *,
    context: EntryKey,
    errors: list[str],
) -> None:
    for field in fields:
        if field not in reference:
            continue
        ref_value = float(reference[field])
        try:
            actual_value = float(actual.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            actual_value = 0.0
        delta = abs(actual_value - ref_value)
        if delta > tolerance:
            errors.append(
                f"{context.algorithm} N={context.num_nodes} T={context.packet_interval_s}: "
                f"écart {field}={delta:.3f} (> {tolerance})"
            )


def _check_cluster_targets(
    actual: Mapping[str, object],
    tolerance: float,
    *,
    context: EntryKey,
    errors: list[str],
) -> None:
    ratios = actual.get("cluster_der_ratio", {})
    if not isinstance(ratios, Mapping):
        return
    for cluster_id, raw_ratio in ratios.items():
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            continue
        if ratio + tolerance < 1.0:
            errors.append(
                f"{context.algorithm} N={context.num_nodes} T={context.packet_interval_s}: "
                f"cluster {cluster_id} sous la cible (ratio={ratio:.3f})"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        type=Path,
        default=DEFAULT_SERIES_PATH,
        help="Chemin vers le fichier JSON produit par le mode validation",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Fichier JSON de référence (valeurs issues du papier ou mises à jour internes)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolérance maximale autorisée sur les métriques normalisées",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        series_payload = _load_payload(args.series)
        reference_payload = _load_payload(args.reference)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    series_index = _index_entries(series_payload)
    reference_index = _index_entries(reference_payload)
    errors: list[str] = []

    for key, reference_entry in reference_index.items():
        actual_entry = series_index.get(key)
        if actual_entry is None:
            errors.append(
                f"Valeur manquante pour {key.algorithm} N={key.num_nodes} T={key.packet_interval_s}"
            )
            continue
        _compare_numeric(
            actual_entry,
            reference_entry,
            fields=("der_normalized", "throughput_normalized", "gap_normalized"),
            tolerance=args.tolerance,
            context=key,
            errors=errors,
        )
        _check_cluster_targets(actual_entry, args.tolerance, context=key, errors=errors)

    if errors:
        for line in errors:
            print(line)
        return 1
    print("Validation QoS : séries conformes aux références.")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    sys.exit(main())
