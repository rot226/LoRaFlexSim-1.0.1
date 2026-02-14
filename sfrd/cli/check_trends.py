"""Script QA: contrôle des tendances PDR/throughput sur CSV agrégés finaux."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

_REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("SNIR_OFF/pdr_results.csv", "pdr"),
    ("SNIR_OFF/throughput_results.csv", "throughput_packets_per_s"),
    ("SNIR_ON/pdr_results.csv", "pdr"),
    ("SNIR_ON/throughput_results.csv", "throughput_packets_per_s"),
)


@dataclass(frozen=True)
class MetricRow:
    network_size: int
    algorithm: str
    snir: str
    value: float


@dataclass(frozen=True)
class Anomaly:
    rule: str
    severity: str
    metric: str
    algorithm: str
    snir: str
    network_size: str
    details: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vérifie les tendances QA des résultats finaux (PDR/throughput) "
            "et génère un rapport texte + CSV d'anomalies."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("sfrd/output"),
        help="Dossier racine contenant SNIR_OFF/ et SNIR_ON/.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Dossier de sortie des rapports (défaut: <output-root>/qa).",
    )
    parser.add_argument(
        "--global-increase-tolerance",
        type=float,
        default=0.01,
        help=(
            "Tolérance absolue autorisée pour une hausse locale lors de la "
            "vérification de décroissance globale (ex: 0.01 pour PDR)."
        ),
    )
    parser.add_argument(
        "--snir-separation-tolerance",
        type=float,
        default=0.0,
        help=(
            "Tolérance absolue pour accepter OFF <= ON + tolérance. "
            "Au-delà: anomalie de séparation OFF/ON."
        ),
    )
    parser.add_argument(
        "--rupture-threshold",
        type=float,
        default=0.35,
        help=(
            "Seuil relatif de rupture entre tailles adjacentes; "
            "|v2-v1|/max(|v1|,eps) > seuil => anomalie."
        ),
    )
    parser.add_argument(
        "--strong-rank-inversion",
        type=float,
        default=1.5,
        help=(
            "Variation minimale de rang moyen entre tailles adjacentes "
            "pour signaler une inversion forte."
        ),
    )
    return parser.parse_args()


def _read_metric_rows(csv_path: Path, metric_name: str) -> list[MetricRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV manquant: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[MetricRow] = []
        for line_idx, row in enumerate(reader, start=2):
            try:
                rows.append(
                    MetricRow(
                        network_size=int((row.get("network_size") or "").strip()),
                        algorithm=(row.get("algorithm") or "").strip(),
                        snir=(row.get("snir") or "").strip().upper(),
                        value=float((row.get(metric_name) or "").strip()),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_idx} valeur invalide pour {metric_name}"
                ) from exc

    if not rows:
        raise ValueError(f"CSV vide: {csv_path}")
    return rows


def _group_by_algo_snir(rows: Iterable[MetricRow]) -> dict[tuple[str, str], list[MetricRow]]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.algorithm, row.snir), []).append(row)
    return grouped


def _group_by_size_algo(rows: Iterable[MetricRow]) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row.network_size, row.algorithm), {})[row.snir] = row.value
    return grouped


def _detect_global_decrease(
    metric: str,
    rows: list[MetricRow],
    tolerance: float,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for (algorithm, snir), subset in _group_by_algo_snir(rows).items():
        sorted_rows = sorted(subset, key=lambda item: item.network_size)
        for previous, current in zip(sorted_rows, sorted_rows[1:]):
            if current.value - previous.value > tolerance:
                anomalies.append(
                    Anomaly(
                        rule="global_monotonic_decrease",
                        severity="warning",
                        metric=metric,
                        algorithm=algorithm,
                        snir=snir,
                        network_size=f"{previous.network_size}->{current.network_size}",
                        details=(
                            f"Hausse observée {previous.value:.6f}->{current.value:.6f} "
                            f"(tol={tolerance:.6f})"
                        ),
                    )
                )
    return anomalies


def _detect_snir_separation(
    metric: str,
    rows: list[MetricRow],
    tolerance: float,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for (network_size, algorithm), values_by_snir in _group_by_size_algo(rows).items():
        if "OFF" not in values_by_snir or "ON" not in values_by_snir:
            continue
        off_value = values_by_snir["OFF"]
        on_value = values_by_snir["ON"]
        if off_value > on_value + tolerance:
            anomalies.append(
                Anomaly(
                    rule="snir_off_vs_on",
                    severity="warning",
                    metric=metric,
                    algorithm=algorithm,
                    snir="OFF/ON",
                    network_size=str(network_size),
                    details=(
                        f"OFF={off_value:.6f} > ON={on_value:.6f} + tol={tolerance:.6f}"
                    ),
                )
            )
    return anomalies


def _detect_adjacent_breaks(
    metric: str,
    rows: list[MetricRow],
    threshold: float,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    epsilon = 1e-9
    for (algorithm, snir), subset in _group_by_algo_snir(rows).items():
        sorted_rows = sorted(subset, key=lambda item: item.network_size)
        for previous, current in zip(sorted_rows, sorted_rows[1:]):
            rel_jump = abs(current.value - previous.value) / max(abs(previous.value), epsilon)
            if rel_jump > threshold:
                anomalies.append(
                    Anomaly(
                        rule="adjacent_size_break",
                        severity="warning",
                        metric=metric,
                        algorithm=algorithm,
                        snir=snir,
                        network_size=f"{previous.network_size}->{current.network_size}",
                        details=(
                            f"Saut relatif={rel_jump:.4f} (seuil={threshold:.4f}), "
                            f"valeurs {previous.value:.6f}->{current.value:.6f}"
                        ),
                    )
                )
    return anomalies


def _average_ranks_by_size(
    pdr_rows: list[MetricRow],
    throughput_rows: list[MetricRow],
) -> dict[int, dict[str, float]]:
    def _mean_by_size_algo(rows: list[MetricRow]) -> dict[tuple[int, str], float]:
        buckets: dict[tuple[int, str], list[float]] = {}
        for row in rows:
            buckets.setdefault((row.network_size, row.algorithm), []).append(row.value)
        return {key: mean(values) for key, values in buckets.items()}

    pdr_means = _mean_by_size_algo(pdr_rows)
    thr_means = _mean_by_size_algo(throughput_rows)
    sizes = sorted({size for size, _ in pdr_means} | {size for size, _ in thr_means})

    size_to_rank: dict[int, dict[str, float]] = {}
    for size in sizes:
        algos = sorted(
            {algo for (candidate_size, algo) in pdr_means if candidate_size == size}
            | {algo for (candidate_size, algo) in thr_means if candidate_size == size}
        )
        ranking_scores: dict[str, list[float]] = {algo: [] for algo in algos}

        def _assign_rank(values: dict[str, float]) -> None:
            ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
            for rank, (algo, _) in enumerate(ordered, start=1):
                ranking_scores[algo].append(float(rank))

        pdr_values = {algo: value for (candidate_size, algo), value in pdr_means.items() if candidate_size == size}
        thr_values = {algo: value for (candidate_size, algo), value in thr_means.items() if candidate_size == size}
        if pdr_values:
            _assign_rank(pdr_values)
        if thr_values:
            _assign_rank(thr_values)

        size_to_rank[size] = {
            algo: mean(ranks) for algo, ranks in ranking_scores.items() if ranks
        }

    return size_to_rank


def _detect_rank_inversions(
    pdr_rows: list[MetricRow],
    throughput_rows: list[MetricRow],
    threshold: float,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    ranks = _average_ranks_by_size(pdr_rows, throughput_rows)
    sizes = sorted(ranks)
    for prev_size, curr_size in zip(sizes, sizes[1:]):
        prev_ranks = ranks[prev_size]
        curr_ranks = ranks[curr_size]
        shared_algorithms = sorted(set(prev_ranks) & set(curr_ranks))
        for algorithm in shared_algorithms:
            diff = abs(curr_ranks[algorithm] - prev_ranks[algorithm])
            if diff >= threshold:
                anomalies.append(
                    Anomaly(
                        rule="strong_rank_inversion",
                        severity="info",
                        metric="pdr+throughput",
                        algorithm=algorithm,
                        snir="MIXED",
                        network_size=f"{prev_size}->{curr_size}",
                        details=(
                            f"Rang moyen {prev_ranks[algorithm]:.2f}->{curr_ranks[algorithm]:.2f} "
                            f"(Δ={diff:.2f}, seuil={threshold:.2f})"
                        ),
                    )
                )
    return anomalies


def _write_reports(report_dir: Path, anomalies: list[Anomaly]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path = report_dir / "trend_anomalies.txt"
    csv_path = report_dir / "trend_anomalies.csv"

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Rapport QA des tendances (PDR/throughput)\n")
        handle.write("=" * 44 + "\n\n")
        handle.write(f"Nombre d'anomalies: {len(anomalies)}\n\n")
        if not anomalies:
            handle.write("Aucune anomalie détectée.\n")
        else:
            by_rule: dict[str, list[Anomaly]] = {}
            for anomaly in anomalies:
                by_rule.setdefault(anomaly.rule, []).append(anomaly)
            for rule, rule_anomalies in sorted(by_rule.items()):
                handle.write(f"- {rule}: {len(rule_anomalies)}\n")
            handle.write("\nDétail:\n")
            for idx, anomaly in enumerate(anomalies, start=1):
                handle.write(
                    f"{idx:03d}. [{anomaly.severity}] {anomaly.rule} | metric={anomaly.metric} "
                    f"| algo={anomaly.algorithm} | snir={anomaly.snir} "
                    f"| size={anomaly.network_size} | {anomaly.details}\n"
                )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rule",
                "severity",
                "metric",
                "algorithm",
                "snir",
                "network_size",
                "details",
            ],
        )
        writer.writeheader()
        for anomaly in anomalies:
            writer.writerow(
                {
                    "rule": anomaly.rule,
                    "severity": anomaly.severity,
                    "metric": anomaly.metric,
                    "algorithm": anomaly.algorithm,
                    "snir": anomaly.snir,
                    "network_size": anomaly.network_size,
                    "details": anomaly.details,
                }
            )

    return txt_path, csv_path


def main() -> None:
    args = _parse_args()
    output_root: Path = args.output_root
    report_dir = args.report_dir or (output_root / "qa")

    metric_rows: dict[str, list[MetricRow]] = {}
    for relative_path, metric_name in _REQUIRED_FILES:
        csv_path = output_root / relative_path
        metric_rows[f"{relative_path}:{metric_name}"] = _read_metric_rows(csv_path, metric_name)

    pdr_rows = [
        *metric_rows["SNIR_OFF/pdr_results.csv:pdr"],
        *metric_rows["SNIR_ON/pdr_results.csv:pdr"],
    ]
    throughput_rows = [
        *metric_rows["SNIR_OFF/throughput_results.csv:throughput_packets_per_s"],
        *metric_rows["SNIR_ON/throughput_results.csv:throughput_packets_per_s"],
    ]

    anomalies: list[Anomaly] = []
    anomalies.extend(_detect_global_decrease("pdr", pdr_rows, args.global_increase_tolerance))
    anomalies.extend(
        _detect_global_decrease(
            "throughput_packets_per_s",
            throughput_rows,
            args.global_increase_tolerance,
        )
    )
    anomalies.extend(_detect_snir_separation("pdr", pdr_rows, args.snir_separation_tolerance))
    anomalies.extend(
        _detect_snir_separation(
            "throughput_packets_per_s",
            throughput_rows,
            args.snir_separation_tolerance,
        )
    )
    anomalies.extend(_detect_adjacent_breaks("pdr", pdr_rows, args.rupture_threshold))
    anomalies.extend(
        _detect_adjacent_breaks(
            "throughput_packets_per_s",
            throughput_rows,
            args.rupture_threshold,
        )
    )
    anomalies.extend(
        _detect_rank_inversions(
            pdr_rows,
            throughput_rows,
            args.strong_rank_inversion,
        )
    )

    txt_path, csv_path = _write_reports(report_dir, anomalies)
    print(f"Rapport texte: {txt_path}")
    print(f"Rapport CSV: {csv_path}")
    print(f"Anomalies détectées: {len(anomalies)}")


if __name__ == "__main__":
    main()
