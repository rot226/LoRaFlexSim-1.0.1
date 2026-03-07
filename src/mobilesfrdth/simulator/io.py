"""Entrées/sorties CSV pour runs et agrégations mobilesfrdth."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import der, jain_fairness, outage_ratio, pdr, throughput

SCENARIO_ID_COLUMNS = ["N", "speed", "mobility_model", "mode", "algo", "gateways", "sigma", "seed", "rep"]
EVENT_COLUMNS = [
    *SCENARIO_ID_COLUMNS,
    "run_id",
    "event_idx",
    "time_s",
    "event_type",
    "node_id",
    "sf",
    "sinr_db",
    "success",
    "delivered",
    "payload_bytes",
    "airtime_s",
    "outage",
    "switch_count",
]
NODE_TIMESERIES_COLUMNS = [
    *SCENARIO_ID_COLUMNS,
    "run_id",
    "bin_start_s",
    "bin_end_s",
    "node_id",
    "tx_count",
    "success_count",
    "delivery_ratio",
    "throughput_bps",
    "mean_sinr_db",
    "airtime_s",
    "outage_count",
    "switch_count",
]
SUMMARY_COLUMNS = [
    *SCENARIO_ID_COLUMNS,
    "run_id",
    "duration_s",
    "node_count",
    "tx_count",
    "success_count",
    "generated_packets",
    "delivered_bytes",
    "pdr",
    "der",
    "throughput_bps",
    "Tc_s",
    "jain_fairness",
    "airtime_total_s",
    "airtime_mean_per_node_s",
    "outage_ratio",
    "switch_count",
]


def _scenario_row(run_config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: run_config.get(key, "") for key in SCENARIO_ID_COLUMNS}


def _coerce_event(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    payload: dict[str, Any] = {
        "time_s": getattr(event, "time_s", 0.0),
        "event_type": getattr(event, "kind", "uplink"),
        "node_id": getattr(event, "node_id", -1),
    }
    for field in ("sf", "sinr_db", "success", "delivered", "payload_bytes", "airtime_s", "outage", "switch_count"):
        if hasattr(event, field):
            payload[field] = getattr(event, field)
    return payload


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_run_outputs(
    *,
    output_root: Path,
    run_id: str,
    run_config: Mapping[str, Any],
    events: Iterable[Any],
    duration_s: float,
    time_bin_s: float = 10.0,
) -> dict[str, Path]:
    """Écrit les artefacts d'un run dans ``results/<run_id>/``."""

    if duration_s <= 0:
        raise ValueError("duration_s doit être > 0")
    if time_bin_s <= 0:
        raise ValueError("time_bin_s doit être > 0")

    run_dir = output_root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(dict(run_config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    scenario = _scenario_row(run_config)
    event_rows: list[dict[str, Any]] = []
    bins: dict[tuple[int, int], dict[str, Any]] = {}
    node_successes: Counter[int] = Counter()
    node_airtime: Counter[int] = Counter()

    tx_count = 0
    success_count = 0
    generated_packets = 0
    delivered_bytes = 0
    outage_events = 0
    total_switch_count = 0
    tc_candidate = math.inf

    for idx, item in enumerate(events):
        event = _coerce_event(item)
        event_type = str(event.get("event_type", "uplink"))
        time_s = float(event.get("time_s", 0.0))
        node_id = int(event.get("node_id", -1))
        sf = int(event.get("sf", 7) or 7)
        sinr_db = float(event.get("sinr_db", 0.0) or 0.0)
        success = int(bool(event.get("success", event_type == "uplink")))
        delivered = int(bool(event.get("delivered", success)))
        payload_bytes = int(event.get("payload_bytes", 0) or 0)
        airtime_s = float(event.get("airtime_s", 0.0) or 0.0)
        outage = int(bool(event.get("outage", not success)))
        switch_count = int(event.get("switch_count", 0) or 0)

        row = {
            **scenario,
            "run_id": run_id,
            "event_idx": idx,
            "time_s": time_s,
            "event_type": event_type,
            "node_id": node_id,
            "sf": sf,
            "sinr_db": sinr_db,
            "success": success,
            "delivered": delivered,
            "payload_bytes": payload_bytes,
            "airtime_s": airtime_s,
            "outage": outage,
            "switch_count": switch_count,
        }
        event_rows.append(row)

        if event_type == "uplink":
            tx_count += 1
            generated_packets += 1
            success_count += success
            delivered_bytes += payload_bytes if delivered else 0
            outage_events += outage
            total_switch_count += switch_count
            node_successes[node_id] += success
            node_airtime[node_id] += airtime_s

            if math.isinf(tc_candidate) and tx_count > 0 and pdr(success_count, tx_count) >= 0.95:
                tc_candidate = time_s

            bin_index = int(time_s // time_bin_s)
            key = (bin_index, node_id)
            if key not in bins:
                bins[key] = {
                    **scenario,
                    "run_id": run_id,
                    "bin_start_s": bin_index * time_bin_s,
                    "bin_end_s": (bin_index + 1) * time_bin_s,
                    "node_id": node_id,
                    "tx_count": 0,
                    "success_count": 0,
                    "sinr_sum": 0.0,
                    "airtime_s": 0.0,
                    "outage_count": 0,
                    "switch_count": 0,
                    "delivered_bytes": 0,
                }
            slot = bins[key]
            slot["tx_count"] += 1
            slot["success_count"] += success
            slot["sinr_sum"] += sinr_db
            slot["airtime_s"] += airtime_s
            slot["outage_count"] += outage
            slot["switch_count"] += switch_count
            slot["delivered_bytes"] += payload_bytes if delivered else 0

    _write_csv(run_dir / "events.csv", EVENT_COLUMNS, event_rows)

    node_timeseries_rows: list[dict[str, Any]] = []
    for (_, _), slot in sorted(bins.items(), key=lambda item: (item[0][0], item[0][1])):
        tx = int(slot["tx_count"])
        success = int(slot["success_count"])
        duration = float(slot["bin_end_s"] - slot["bin_start_s"])
        node_timeseries_rows.append(
            {
                **scenario,
                "run_id": run_id,
                "bin_start_s": slot["bin_start_s"],
                "bin_end_s": slot["bin_end_s"],
                "node_id": slot["node_id"],
                "tx_count": tx,
                "success_count": success,
                "delivery_ratio": der(success, tx),
                "throughput_bps": throughput(int(slot["delivered_bytes"]), duration),
                "mean_sinr_db": slot["sinr_sum"] / max(tx, 1),
                "airtime_s": slot["airtime_s"],
                "outage_count": slot["outage_count"],
                "switch_count": slot["switch_count"],
            }
        )
    _write_csv(run_dir / "node_timeseries.csv", NODE_TIMESERIES_COLUMNS, node_timeseries_rows)

    node_ids = {int(row["node_id"]) for row in event_rows if int(row["node_id"]) >= 0}
    airtime_total = sum(float(v) for v in node_airtime.values())

    summary_row = {
        **scenario,
        "run_id": run_id,
        "duration_s": duration_s,
        "node_count": len(node_ids),
        "tx_count": tx_count,
        "success_count": success_count,
        "generated_packets": generated_packets,
        "delivered_bytes": delivered_bytes,
        "pdr": pdr(success_count, tx_count),
        "der": der(success_count, generated_packets),
        "throughput_bps": throughput(delivered_bytes, duration_s),
        "Tc_s": tc_candidate,
        "jain_fairness": jain_fairness(node_successes.values()),
        "airtime_total_s": airtime_total,
        "airtime_mean_per_node_s": airtime_total / max(len(node_ids), 1),
        "outage_ratio": outage_ratio(outage_events, tx_count),
        "switch_count": total_switch_count,
    }
    _write_csv(run_dir / "summary.csv", SUMMARY_COLUMNS, [summary_row])

    return {
        "run_dir": run_dir,
        "run_config": run_config_path,
        "events": run_dir / "events.csv",
        "node_timeseries": run_dir / "node_timeseries.csv",
        "summary": run_dir / "summary.csv",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _collect_run_dirs(paths: Iterable[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "summary.csv":
            run_dirs.append(path.parent)
            continue
        if path.is_dir() and (path / "summary.csv").is_file():
            run_dirs.append(path)
            continue
        if path.is_dir() and (path / "results").is_dir():
            for candidate in sorted((path / "results").iterdir()):
                if candidate.is_dir() and (candidate / "summary.csv").is_file():
                    run_dirs.append(candidate)
    unique = []
    seen = set()
    for item in run_dirs:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)
    if not unique:
        raise ValueError("Aucun run valide trouvé (summary.csv introuvable).")
    return unique


def aggregate_runs(
    *,
    inputs: Iterable[Path],
    output_root: Path,
    summary_only: bool = False,
    skip_sinr_cdf: bool = False,
    skip_sf_distribution: bool = False,
) -> dict[str, Path]:
    """Agrège des runs et écrit les CSV dans ``aggregates/``."""

    run_dirs = _collect_run_dirs(inputs)
    out_dir = output_root / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary_only:
        skip_sinr_cdf = True
        skip_sf_distribution = True

    summaries: list[dict[str, str]] = []
    sf_counter: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    sinr_values: dict[tuple[str, ...], list[float]] = defaultdict(list)

    factor_columns = ["N", "speed", "mobility_model", "mode", "algo", "gateways", "sigma"]

    total = len(run_dirs)
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"Agrégation des runs: {index}/{total}", end="\r" if index < total else "\n", flush=True)
        summaries.extend(_read_csv(run_dir / "summary.csv"))

        if summary_only:
            continue

        for row in _iter_csv(run_dir / "events.csv"):
            if row.get("event_type") != "uplink":
                continue
            key = tuple(row.get(column, "") for column in factor_columns)
            if not skip_sf_distribution:
                sf_counter[key][row.get("sf", "")] += 1
            if not skip_sinr_cdf:
                sinr_values[key].append(float(row.get("sinr_db", 0.0) or 0.0))

    by_factor: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in summaries:
        by_factor[tuple(row.get(column, "") for column in factor_columns)].append(row)

    metric_by_factor_rows = []
    for key, rows in sorted(by_factor.items()):
        def avg(name: str) -> float:
            values = [float(r.get(name, 0.0) or 0.0) for r in rows]
            return mean(values) if values else 0.0

        metric_by_factor_rows.append(
            {
                **dict(zip(factor_columns, key, strict=False)),
                "num_runs": len(rows),
                "pdr_mean": avg("pdr"),
                "der_mean": avg("der"),
                "throughput_bps_mean": avg("throughput_bps"),
                "Tc_s_mean": avg("Tc_s"),
                "jain_fairness_mean": avg("jain_fairness"),
                "airtime_total_s_mean": avg("airtime_total_s"),
                "outage_ratio_mean": avg("outage_ratio"),
                "switch_count_mean": avg("switch_count"),
            }
        )
    metric_by_factor_path = out_dir / "metric_by_factor.csv"
    _write_csv(metric_by_factor_path, factor_columns + [
        "num_runs",
        "pdr_mean",
        "der_mean",
        "throughput_bps_mean",
        "Tc_s_mean",
        "jain_fairness_mean",
        "airtime_total_s_mean",
        "outage_ratio_mean",
        "switch_count_mean",
    ], metric_by_factor_rows)

    files: dict[str, Path] = {"metric_by_factor": metric_by_factor_path}

    if not skip_sf_distribution:
        distribution_rows = []
        for key, counts in sorted(sf_counter.items()):
            total_count = sum(counts.values())
            for sf, count in sorted(counts.items(), key=lambda item: int(item[0] or 0)):
                distribution_rows.append(
                    {
                        **dict(zip(factor_columns, key, strict=False)),
                        "sf": sf,
                        "count": count,
                        "ratio": count / max(total_count, 1),
                    }
                )
        distribution_path = out_dir / "distribution_sf.csv"
        _write_csv(distribution_path, factor_columns + ["sf", "count", "ratio"], distribution_rows)
        files["distribution_sf"] = distribution_path

    convergence_rows = [
        {
            **{column: row.get(column, "") for column in SCENARIO_ID_COLUMNS},
            "run_id": row.get("run_id", ""),
            "Tc_s": row.get("Tc_s", ""),
        }
        for row in summaries
    ]
    convergence_path = out_dir / "convergence_tc.csv"
    _write_csv(convergence_path, SCENARIO_ID_COLUMNS + ["run_id", "Tc_s"], convergence_rows)

    if not skip_sinr_cdf:
        sinr_rows = []
        quantiles = [i / 100 for i in range(101)]
        for key, values in sorted(sinr_values.items()):
            if not values:
                continue
            data = sorted(values)
            n = len(data)
            for q in quantiles:
                index = min(int(round(q * (n - 1))), n - 1)
                sinr_rows.append({**dict(zip(factor_columns, key, strict=False)), "quantile": q, "sinr_db": data[index]})
        sinr_path = out_dir / "sinr_cdf.csv"
        _write_csv(sinr_path, factor_columns + ["quantile", "sinr_db"], sinr_rows)
        files["sinr_cdf"] = sinr_path

    fairness_rows = [
        {
            **{column: row.get(column, "") for column in SCENARIO_ID_COLUMNS},
            "run_id": row.get("run_id", ""),
            "jain_fairness": row.get("jain_fairness", ""),
            "airtime_total_s": row.get("airtime_total_s", ""),
            "switch_count": row.get("switch_count", ""),
        }
        for row in summaries
    ]
    fairness_path = out_dir / "fairness_airtime_switching.csv"
    _write_csv(
        fairness_path,
        SCENARIO_ID_COLUMNS + ["run_id", "jain_fairness", "airtime_total_s", "switch_count"],
        fairness_rows,
    )

    files["convergence_tc"] = convergence_path
    files["fairness_airtime_switching"] = fairness_path

    return files
