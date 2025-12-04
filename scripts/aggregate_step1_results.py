"""Agrège les CSV de l'étape 1 en générant un résumé statistique et un index brut."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.plot_step1_results import STATE_LABELS, _detect_snir
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"


MetricValues = List[float]
ClusterMetrics = Dict[int, MetricValues]
Record = Dict[str, Any]


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return fmean(values), pstdev(values)


def _load_records(results_dir: Path, strict_snir: bool) -> Tuple[List[Record], List[int]]:
    records: List[Record] = []
    cluster_ids: set[int] = set()
    if not results_dir.exists():
        return records, sorted(cluster_ids)

    run_id = 1
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cluster_pdr: Dict[int, float] = {}
                for key, value in row.items():
                    if key.startswith("qos_cluster_pdr__"):
                        cid = int(key.split("__")[-1])
                        cluster_ids.add(cid)
                        cluster_pdr[cid] = _parse_float(value)

                snir_flag = _detect_snir(row, csv_path)
                if strict_snir and snir_flag is None:
                    raise ValueError(
                        f"Impossible de déterminer l'état SNIR pour {csv_path}; utilisez des fichiers explicites."
                    )

                record: Record = {
                    "run_id": run_id,
                    "csv_path": csv_path,
                    "algorithm": row.get("algorithm", csv_path.parent.name),
                    "num_nodes": _parse_int(row.get("num_nodes")),
                    "packet_interval_s": _parse_float(row.get("packet_interval_s")),
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "collisions": _parse_int(row.get("collisions")),
                    "jain_index": _parse_float(row.get("jain_index")),
                    "throughput_bps": _parse_float(row.get("throughput_bps")),
                    "cluster_pdr": cluster_pdr,
                    "use_snir": snir_flag,
                    "snir_state": STATE_LABELS.get(snir_flag, "snir_unknown"),
                }
                records.append(record)
                run_id += 1
    return records, sorted(cluster_ids)


def _group_records(records: Iterable[Record]) -> Dict[Tuple[Any, ...], List[Record]]:
    groups: Dict[Tuple[Any, ...], List[Record]] = defaultdict(list)
    for record in records:
        key = (
            record.get("algorithm"),
            record.get("snir_state"),
            record.get("num_nodes"),
            record.get("packet_interval_s"),
        )
        groups[key].append(record)
    return groups


def _write_summary(
    output_path: Path, groups: Mapping[Tuple[Any, ...], List[Record]], cluster_ids: Sequence[int]
) -> None:
    base_headers = [
        "algorithm",
        "snir_state",
        "num_nodes",
        "packet_interval_s",
        "PDR_mean",
        "PDR_std",
        "DER_mean",
        "DER_std",
        "collisions_mean",
        "collisions_std",
        "jain_index_mean",
        "jain_index_std",
        "throughput_bps_mean",
        "throughput_bps_std",
    ]
    cluster_headers = []
    for cid in cluster_ids:
        cluster_headers.extend([f"cluster_pdr_{cid}_mean", f"cluster_pdr_{cid}_std"])
    headers = base_headers + cluster_headers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for key, items in sorted(groups.items()):
            summary: Dict[str, Any] = {
                "algorithm": key[0],
                "snir_state": key[1],
                "num_nodes": key[2],
                "packet_interval_s": key[3],
            }
            for metric in ("PDR", "DER", "collisions", "jain_index", "throughput_bps"):
                values = [float(r.get(metric, 0.0)) for r in items]
                mean, std = _mean_std(values)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std

            for cid in cluster_ids:
                values = [r.get("cluster_pdr", {}).get(cid) for r in items]
                filtered = [v for v in values if v is not None]
                mean, std = _mean_std(filtered)
                summary[f"cluster_pdr_{cid}_mean"] = mean
                summary[f"cluster_pdr_{cid}_std"] = std
            writer.writerow(summary)


def _write_raw_index(output_path: Path, records: Iterable[Record], cluster_ids: Sequence[int]) -> None:
    base_headers = [
        "run_id",
        "algorithm",
        "snir_state",
        "use_snir",
        "num_nodes",
        "packet_interval_s",
        "csv_path",
        "PDR",
        "DER",
        "collisions",
        "jain_index",
        "throughput_bps",
    ]
    cluster_headers = [f"cluster_pdr_{cid}" for cid in cluster_ids]
    headers = base_headers + cluster_headers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in base_headers}
            # Normalise les chemins pour l'index
            path = record.get("csv_path")
            if isinstance(path, Path):
                try:
                    row["csv_path"] = path.relative_to(ROOT_DIR)
                except ValueError:
                    row["csv_path"] = str(path)
            for cid in cluster_ids:
                row[f"cluster_pdr_{cid}"] = record.get("cluster_pdr", {}).get(cid)
            writer.writerow(row)


def aggregate_step1_results(results_dir: Path, strict_snir_detection: bool) -> None:
    records, cluster_ids = _load_records(results_dir, strict_snir_detection)
    if not records:
        print(f"Aucun CSV trouvé dans {results_dir}")
        return

    groups = _group_records(records)
    summary_path = results_dir / "summary.csv"
    raw_path = results_dir / "raw_index.csv"

    _write_summary(summary_path, groups, cluster_ids)
    _write_raw_index(raw_path, records, cluster_ids)

    print(f"Résumé écrit dans {summary_path}")
    print(f"Index brut écrit dans {raw_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant les CSV de l'étape 1",
    )
    parser.add_argument(
        "--strict-snir-detection",
        action="store_true",
        help="Impose la détection explicite de l'état SNIR (on/off) en utilisant la logique des figures",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    aggregate_step1_results(args.results_dir, args.strict_snir_detection)


if __name__ == "__main__":
    main()
