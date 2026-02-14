"""Point d'entrée module: agrégation des résultats."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(float(value.strip()))
    raise ValueError(f"Valeur entière invalide: {value!r}")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"Valeur numérique invalide: {value!r}")


def _normalize_snir(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"on", "snir_on", "true", "1"}:
        return "ON"
    if normalized in {"off", "snir_off", "false", "0"}:
        return "OFF"
    return str(value or "").strip().upper()


def _extract_metric(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] not in (None, ""):
            return _to_float(metrics[key])
    return 0.0


def _extract_sf_counts(metrics: dict[str, Any], network_size: int) -> dict[int, float]:
    raw = (
        metrics.get("sf_distribution_counts")
        or metrics.get("sf_distribution")
        or metrics.get("sf_counts")
        or {}
    )
    if not isinstance(raw, dict):
        return {}

    parsed: dict[int, float] = {}
    for sf_key, raw_value in raw.items():
        try:
            sf = _to_int(sf_key)
            value = _to_float(raw_value)
        except (ValueError, TypeError):
            continue
        parsed[sf] = value

    total = sum(parsed.values())
    if total <= 0:
        return parsed

    if total <= 1.000001 and network_size > 0:
        return {sf: value * network_size for sf, value in parsed.items()}

    return parsed


def _extract_rewards(summary: dict[str, Any]) -> Iterable[tuple[int, float]]:
    metrics = summary.get("metrics", {})
    candidates = [
        summary.get("learning_curve"),
        summary.get("rewards"),
        metrics.get("learning_curve"),
        metrics.get("ucb_learning_curve"),
        metrics.get("rewards"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            for idx, item in enumerate(candidate, start=1):
                if isinstance(item, dict):
                    ep = item.get("episode", idx)
                    reward = item.get("reward", 0.0)
                else:
                    ep = idx
                    reward = item
                try:
                    yield _to_int(ep), _to_float(reward)
                except (ValueError, TypeError):
                    continue
            return

        if isinstance(candidate, dict):
            for ep, reward in candidate.items():
                try:
                    yield _to_int(ep), _to_float(reward)
                except (ValueError, TypeError):
                    continue
            return


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_logs(logs_root: str | Path) -> Path:
    """Agrège les fichiers ``campaign_summary.json`` en sorties CSV dédiées."""

    root = Path(logs_root)
    summaries = sorted(root.glob("SNIR_*/ns_*/algo_*/seed_*/campaign_summary.json"))

    metric_sums: dict[tuple[int, str, str], dict[str, float]] = defaultdict(
        lambda: {"pdr": 0.0, "throughput": 0.0, "energy": 0.0, "count": 0.0}
    )
    sf_sums: dict[tuple[int, str, str, int], dict[str, float]] = defaultdict(
        lambda: {"count_sum": 0.0, "replications": 0.0}
    )
    rewards_by_episode: dict[int, list[float]] = defaultdict(list)

    for summary_path in summaries:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = data.get("contract", {})
        metrics = data.get("metrics", {})

        network_size = _to_int(contract.get("network_size", 0))
        algorithm = str(contract.get("algorithm", "")).strip()
        snir = _normalize_snir(contract.get("snir_mode", ""))

        key = (network_size, algorithm, snir)
        metric_sums[key]["pdr"] += _extract_metric(metrics, "pdr", "PDR")
        metric_sums[key]["throughput"] += _extract_metric(
            metrics,
            "throughput_packets_per_s",
            "throughput_pps",
            "throughput_bps",
        )
        metric_sums[key]["energy"] += _extract_metric(
            metrics,
            "energy_joule_per_packet",
            "energy_per_packet",
            "energy",
        )
        metric_sums[key]["count"] += 1.0

        sf_counts = _extract_sf_counts(metrics, network_size)
        for sf, value in sf_counts.items():
            sf_key = (network_size, algorithm, snir, sf)
            sf_sums[sf_key]["count_sum"] += value
            sf_sums[sf_key]["replications"] += 1.0

        for episode, reward in _extract_rewards(data):
            rewards_by_episode[episode].append(reward)

    pdr_rows: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []

    for network_size, algorithm, snir in sorted(metric_sums):
        values = metric_sums[(network_size, algorithm, snir)]
        replications = values["count"] or 1.0
        pdr_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "pdr": values["pdr"] / replications,
            }
        )
        throughput_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "throughput_packets_per_s": values["throughput"] / replications,
            }
        )
        energy_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "energy_joule_per_packet": values["energy"] / replications,
            }
        )

    sf_rows: list[dict[str, Any]] = []
    for network_size, algorithm, snir, sf in sorted(sf_sums):
        values = sf_sums[(network_size, algorithm, snir, sf)]
        replications = values["replications"] or 1.0
        sf_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "sf": sf,
                "count": values["count_sum"] / replications,
            }
        )

    learning_rows: list[dict[str, Any]] = []
    for episode in sorted(rewards_by_episode):
        rewards = rewards_by_episode[episode]
        if not rewards:
            continue
        learning_rows.append(
            {
                "episode": episode,
                "reward": sum(rewards) / len(rewards),
            }
        )

    off_root = root.parent / "output" / "SNIR_OFF"
    on_root = root.parent / "output" / "SNIR_ON"
    output_root = root.parent / "output"

    _write_csv(
        off_root / "pdr_results.csv",
        ["network_size", "algorithm", "snir", "pdr"],
        [row for row in pdr_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "throughput_results.csv",
        ["network_size", "algorithm", "snir", "throughput_packets_per_s"],
        [row for row in throughput_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "energy_results.csv",
        ["network_size", "algorithm", "snir", "energy_joule_per_packet"],
        [row for row in energy_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "sf_distribution.csv",
        ["network_size", "algorithm", "snir", "sf", "count"],
        [row for row in sf_rows if row["snir"] == "OFF"],
    )

    _write_csv(
        on_root / "pdr_results.csv",
        ["network_size", "algorithm", "snir", "pdr"],
        [row for row in pdr_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "throughput_results.csv",
        ["network_size", "algorithm", "snir", "throughput_packets_per_s"],
        [row for row in throughput_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "energy_results.csv",
        ["network_size", "algorithm", "snir", "energy_joule_per_packet"],
        [row for row in energy_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "sf_distribution.csv",
        ["network_size", "algorithm", "snir", "sf", "count"],
        [row for row in sf_rows if row["snir"] == "ON"],
    )

    _write_csv(
        output_root / "learning_curve_ucb.csv",
        ["episode", "reward"],
        learning_rows,
    )

    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agrège les résumés de campagne SFRD.")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Racine des logs (contient SNIR_OFF/SNIR_ON)",
    )
    return parser.parse_args()


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    path = aggregate_logs(args.logs_root)
    print(f"Agrégation écrite: {path}")


if __name__ == "__main__":
    main()
