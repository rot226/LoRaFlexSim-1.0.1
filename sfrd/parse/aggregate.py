"""Point d'entrée module: agrégation des résultats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def aggregate_logs(logs_root: str | Path) -> Path:
    """Agrège les fichiers ``campaign_summary.json`` en CSV + JSON."""

    root = Path(logs_root)
    summaries = sorted(root.glob("SNIR_*/ns_*/algo_*/seed_*/campaign_summary.json"))

    rows: list[dict[str, object]] = []
    for summary_path in summaries:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = data.get("contract", {})
        metrics = data.get("metrics", {})

        rows.append(
            {
                "snir_mode": contract.get("snir_mode", ""),
                "network_size": contract.get("network_size", ""),
                "algorithm": contract.get("algorithm", ""),
                "seed": contract.get("seed", ""),
                "warmup_s": contract.get("warmup_s", ""),
                "pdr": metrics.get("pdr", ""),
                "throughput_bps": metrics.get("throughput_bps", ""),
                "collisions": metrics.get("collisions", ""),
                "tx_attempted": metrics.get("tx_attempted", ""),
                "rx_delivered": metrics.get("rx_delivered", ""),
                "summary_path": str(summary_path),
            }
        )

    output_json = root / "aggregate_runs.json"
    output_csv = root / "aggregate_runs.csv"

    output_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    headers = [
        "snir_mode",
        "network_size",
        "algorithm",
        "seed",
        "warmup_s",
        "pdr",
        "throughput_bps",
        "collisions",
        "tx_attempted",
        "rx_delivered",
        "summary_path",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return output_json


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
