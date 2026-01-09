"""Normalise les sorties Step 2 vers results/step2/raw et results/step2/agg."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "experiments" / "ucb1"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "step2"

STATE_LABELS = {True: "snir_on", False: "snir_off", None: "snir_unknown"}


DecisionRow = Dict[str, Any]
MetricsRow = Dict[str, Any]


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _normalize_algorithm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mixra_opt": {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"},
        "mixra_h": {"mixra_h", "mixrah", "mixra-h", "mixra h"},
        "adr_max": {"adr_max", "adr-max", "adr_max", "adr max", "adr-max"},
        "adr_avg": {"adr_avg", "adr-avg", "adr_avg", "adr avg", "adr-avg"},
        "ucb1": {"ucb1", "ucb-1"},
    }
    for canonical, names in aliases.items():
        if normalized in names:
            return canonical
    return text


def _snir_state_from_path(path: Path) -> str | None:
    lowered = path.as_posix().lower()
    if "snir_on" in lowered or "sniron" in lowered:
        return STATE_LABELS.get(True)
    if "snir_off" in lowered or "sniroff" in lowered:
        return STATE_LABELS.get(False)
    return None


def _snir_state_from_row(row: Mapping[str, Any]) -> str | None:
    raw = row.get("snir_state")
    if raw is not None and str(raw).strip() != "":
        normalized = str(raw).strip().lower()
        if normalized in {"snir_on", "on", "true", "1", "yes", "y"}:
            return "snir_on"
        if normalized in {"snir_off", "off", "false", "0", "no", "n"}:
            return "snir_off"
        if normalized in {"snir_unknown", "unknown", "na", "n/a"}:
            return "snir_unknown"
    if "use_snir" in row:
        parsed = _parse_bool(row.get("use_snir"))
    else:
        parsed = _parse_bool(row.get("with_snir"))
    return STATE_LABELS.get(parsed) if parsed is not None else None


def _detect_csv_kind(header: Sequence[str]) -> str | None:
    columns = {name.strip().lower() for name in header}
    if {"episode_idx", "decision_idx"}.issubset(columns):
        return "decision"
    if "reward_mean" in columns and "pdr" in columns:
        return "metrics"
    return None


def _load_csv(path: Path) -> Tuple[str | None, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        kind = _detect_csv_kind(reader.fieldnames or [])
        if kind is None:
            return None, []
        return kind, [row for row in reader]


def _collect_inputs(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted({path for path in input_dir.rglob("*.csv") if path.is_file()})


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean = fmean(values)
    std = pstdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, margin


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_decision_rows(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    run_id: int,
    snir_state: str,
    algorithm: str,
) -> List[DecisionRow]:
    rows: List[DecisionRow] = []
    for row in raw_rows:
        episode_idx = _parse_int(row.get("episode_idx"))
        decision_idx = _parse_int(row.get("decision_idx"))
        round_idx = episode_idx if episode_idx is not None else decision_idx
        rows.append(
            {
                "run_id": run_id,
                "snir_state": snir_state,
                "algorithm": algorithm,
                "episode_idx": episode_idx,
                "decision_idx": decision_idx,
                "round_idx": round_idx,
                "time_s": _parse_float(row.get("time_s"), 0.0) or 0.0,
                "reward": _parse_float(row.get("reward"), 0.0) or 0.0,
                "pdr": _parse_float(row.get("pdr"), 0.0) or 0.0,
                "throughput": _parse_float(row.get("throughput"), 0.0) or 0.0,
                "snir_db": _parse_float(row.get("snir_db")),
                "sf": _parse_int(row.get("sf")),
                "tx_power": _parse_float(row.get("tx_power")),
                "policy": (row.get("policy") or "").strip() or None,
                "cluster": _parse_int(row.get("cluster")),
                "num_nodes": _parse_int(row.get("num_nodes")),
                "packet_interval_s": _parse_float(row.get("packet_interval_s")),
                "energy_j": _parse_float(row.get("energy_j")),
            }
        )
    return rows


def _normalize_metric_rows(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    run_id: int,
    snir_state: str,
    algorithm: str,
) -> List[MetricsRow]:
    rows: List[MetricsRow] = []
    for row in raw_rows:
        rows.append(
            {
                "run_id": run_id,
                "snir_state": snir_state,
                "algorithm": algorithm,
                "cluster": _parse_int(row.get("cluster")),
                "num_nodes": _parse_int(row.get("num_nodes")),
                "sf": _parse_float(row.get("sf")),
                "reward_mean": _parse_float(row.get("reward_mean")),
                "reward_variance": _parse_float(row.get("reward_variance")),
                "der": _parse_float(row.get("der")),
                "pdr": _parse_float(row.get("pdr")),
                "snir_avg": _parse_float(row.get("snir_avg")),
                "success_rate": _parse_float(row.get("success_rate")),
            }
        )
    return rows


def _aggregate_decisions(decisions: Sequence[DecisionRow]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    performance_rows: List[Dict[str, Any]] = []
    convergence_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, int], List[DecisionRow]] = {}
    convergence_grouped: Dict[Tuple[str, str, int], List[DecisionRow]] = {}

    for row in decisions:
        round_idx = row.get("round_idx")
        if round_idx is not None:
            key = (row["snir_state"], row["algorithm"], int(round_idx))
            grouped.setdefault(key, []).append(row)
        episode_idx = row.get("episode_idx")
        if episode_idx is not None:
            conv_key = (row["snir_state"], row["algorithm"], int(episode_idx))
            convergence_grouped.setdefault(conv_key, []).append(row)

    for (snir_state, algorithm, round_idx), items in sorted(grouped.items()):
        reward_vals = [float(r["reward"]) for r in items]
        pdr_vals = [float(r["pdr"]) for r in items]
        throughput_vals = [float(r["throughput"]) for r in items]
        reward_mean, reward_ci = _mean_ci(reward_vals)
        pdr_mean, pdr_ci = _mean_ci(pdr_vals)
        throughput_mean, throughput_ci = _mean_ci(throughput_vals)
        performance_rows.append(
            {
                "snir_state": snir_state,
                "algorithm": algorithm,
                "round_idx": round_idx,
                "reward_mean": reward_mean,
                "reward_ci95": reward_ci,
                "pdr_mean": pdr_mean,
                "pdr_ci95": pdr_ci,
                "throughput_mean": throughput_mean,
                "throughput_ci95": throughput_ci,
                "samples": len(items),
            }
        )

    for (snir_state, algorithm, episode_idx), items in sorted(convergence_grouped.items()):
        reward_vals = [float(r["reward"]) for r in items]
        pdr_vals = [float(r["pdr"]) for r in items]
        throughput_vals = [float(r["throughput"]) for r in items]
        reward_mean, reward_ci = _mean_ci(reward_vals)
        pdr_mean, pdr_ci = _mean_ci(pdr_vals)
        throughput_mean, throughput_ci = _mean_ci(throughput_vals)
        convergence_rows.append(
            {
                "snir_state": snir_state,
                "algorithm": algorithm,
                "episode_idx": episode_idx,
                "reward_mean": reward_mean,
                "reward_ci95": reward_ci,
                "pdr_mean": pdr_mean,
                "pdr_ci95": pdr_ci,
                "throughput_mean": throughput_mean,
                "throughput_ci95": throughput_ci,
                "samples": len(items),
            }
        )

    return performance_rows, convergence_rows


def _aggregate_sf_tp(decisions: Sequence[DecisionRow]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, float], int] = {}
    totals: Dict[Tuple[str, str], int] = {}
    rows: List[Dict[str, Any]] = []

    for row in decisions:
        sf = row.get("sf")
        tx_power = row.get("tx_power")
        if sf is None or tx_power is None:
            continue
        snir_state = row["snir_state"]
        algorithm = row["algorithm"]
        key = (snir_state, algorithm, int(sf), float(tx_power))
        grouped[key] = grouped.get(key, 0) + 1
        totals[(snir_state, algorithm)] = totals.get((snir_state, algorithm), 0) + 1

    for (snir_state, algorithm, sf, tx_power), count in sorted(grouped.items()):
        total = totals.get((snir_state, algorithm), 1)
        rows.append(
            {
                "snir_state": snir_state,
                "algorithm": algorithm,
                "sf": sf,
                "tx_power": tx_power,
                "count": count,
                "share": count / total,
            }
        )
    return rows


def run_normalisation(input_dir: Path, output_dir: Path) -> None:
    input_paths = _collect_inputs(input_dir)
    decisions: List[DecisionRow] = []
    metrics: List[MetricsRow] = []
    run_id = 1
    for path in input_paths:
        kind, rows = _load_csv(path)
        if not rows or kind is None:
            continue
        snir_state = _snir_state_from_path(path)
        algorithm = None
        if rows:
            snir_state = _snir_state_from_row(rows[0]) or snir_state or STATE_LABELS.get(None)
            algorithm = _normalize_algorithm(rows[0].get("algorithm"))
        if algorithm is None:
            algorithm = _normalize_algorithm(path.parent.name) or path.parent.name
        if snir_state is None:
            snir_state = STATE_LABELS.get(None)

        if kind == "decision":
            decisions.extend(
                _normalize_decision_rows(rows, run_id=run_id, snir_state=snir_state, algorithm=algorithm)
            )
        elif kind == "metrics":
            metrics.extend(
                _normalize_metric_rows(rows, run_id=run_id, snir_state=snir_state, algorithm=algorithm)
            )
        run_id += 1

    raw_dir = output_dir / "raw"
    agg_dir = output_dir / "agg"
    _ensure_dir(raw_dir)
    _ensure_dir(agg_dir)

    if decisions:
        decision_fields = [
            "run_id",
            "snir_state",
            "algorithm",
            "episode_idx",
            "decision_idx",
            "round_idx",
            "time_s",
            "reward",
            "pdr",
            "throughput",
            "snir_db",
            "sf",
            "tx_power",
            "policy",
            "cluster",
            "num_nodes",
            "packet_interval_s",
            "energy_j",
        ]
        _write_csv(raw_dir / "decisions.csv", decision_fields, decisions)
        performance_rows, convergence_rows = _aggregate_decisions(decisions)
        if performance_rows:
            _write_csv(
                agg_dir / "performance_rounds.csv",
                [
                    "snir_state",
                    "algorithm",
                    "round_idx",
                    "reward_mean",
                    "reward_ci95",
                    "pdr_mean",
                    "pdr_ci95",
                    "throughput_mean",
                    "throughput_ci95",
                    "samples",
                ],
                performance_rows,
            )
        if convergence_rows:
            _write_csv(
                agg_dir / "convergence.csv",
                [
                    "snir_state",
                    "algorithm",
                    "episode_idx",
                    "reward_mean",
                    "reward_ci95",
                    "pdr_mean",
                    "pdr_ci95",
                    "throughput_mean",
                    "throughput_ci95",
                    "samples",
                ],
                convergence_rows,
            )
        sf_rows = _aggregate_sf_tp(decisions)
        if sf_rows:
            _write_csv(
                agg_dir / "sf_tp_distribution.csv",
                ["snir_state", "algorithm", "sf", "tx_power", "count", "share"],
                sf_rows,
            )

    if metrics:
        metrics_fields = [
            "run_id",
            "snir_state",
            "algorithm",
            "cluster",
            "num_nodes",
            "sf",
            "reward_mean",
            "reward_variance",
            "der",
            "pdr",
            "snir_avg",
            "success_rate",
        ]
        _write_csv(raw_dir / "metrics.csv", metrics_fields, metrics)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Répertoire contenant les CSV Step 2 à normaliser.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour results/step2.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_normalisation(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
