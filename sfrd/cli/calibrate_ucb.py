"""Mini-campagne de calibration UCB puis sélection d'une config figée."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_metric(rows: list[dict[str, str]], metric: str, *, algo: str = "UCB1") -> float:
    values = [
        float(row[metric])
        for row in rows
        if row.get("algorithm", "").strip().upper() == algo and row.get(metric)
    ]
    return sum(values) / len(values) if values else 0.0


def _last_learning_reward(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda row: int(float(row.get("episode", "0") or 0)))
    return float(ordered[-1].get("reward", 0.0))


def _energy_compromise_score(pdr: float, energy: float) -> float:
    return pdr - 0.2 * energy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-campagne de calibration UCB.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=Path,
        default=[
            Path("sfrd/config/calibration/ucb_candidate_a.json"),
            Path("sfrd/config/calibration/ucb_candidate_b.json"),
        ],
    )
    parser.add_argument("--network-sizes", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--replications", type=int, default=2)
    parser.add_argument("--seeds-base", type=int, default=101)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--logs-root", type=Path, default=Path("sfrd/logs/ucb_calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("sfrd/output/ucb_calibration"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, float | str]] = []

    for candidate in args.candidates:
        name = candidate.stem
        logs_root = args.logs_root / name
        run_cmd = [
            sys.executable,
            "-m",
            "sfrd.cli.run_campaign",
            "--network-sizes",
            *[str(val) for val in args.network_sizes],
            "--replications",
            str(args.replications),
            "--seeds-base",
            str(args.seeds_base),
            "--snir",
            "ON",
            "--algos",
            "UCB",
            "--warmup-s",
            str(args.warmup_s),
            "--logs-root",
            str(logs_root),
            "--ucb-config",
            str(candidate),
            "--force-rerun",
        ]
        _run_command(run_cmd)

        output_root = logs_root.parent / "output"
        pdr_rows = _read_csv_rows(output_root / "SNIR_ON" / "pdr_results.csv")
        throughput_rows = _read_csv_rows(output_root / "SNIR_ON" / "throughput_results.csv")
        energy_rows = _read_csv_rows(output_root / "SNIR_ON" / "energy_results.csv")
        learning_rows = _read_csv_rows(output_root / "learning_curve_ucb.csv")

        pdr_mean = _mean_metric(pdr_rows, "pdr")
        throughput_mean = _mean_metric(throughput_rows, "throughput_packets_per_s")
        energy_mean = _mean_metric(energy_rows, "energy_joule_per_packet")
        reward_last = _last_learning_reward(learning_rows)
        score = _energy_compromise_score(pdr_mean, energy_mean)

        target_dir = args.output_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(output_root, target_dir)

        comparisons.append(
            {
                "candidate": name,
                "config_path": str(candidate),
                "pdr_mean": pdr_mean,
                "throughput_mean": throughput_mean,
                "energy_mean": energy_mean,
                "learning_reward_last": reward_last,
                "compromise_score": score,
                "output_dir": str(target_dir),
            }
        )

    best = max(comparisons, key=lambda row: float(row["compromise_score"]))
    best_config = Path(str(best["config_path"]))
    frozen_path = Path("sfrd/config/ucb_config.json")
    shutil.copyfile(best_config, frozen_path)

    summary = {
        "criterion": "maximize(pdr_mean - 0.2 * energy_mean)",
        "chosen_candidate": best["candidate"],
        "chosen_config": str(best_config),
        "frozen_config_path": str(frozen_path),
        "results": comparisons,
    }
    summary_path = args.output_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Calibration terminée. Config choisie: {best['candidate']}")
    print(f"Résumé: {summary_path}")


if __name__ == "__main__":
    main()
