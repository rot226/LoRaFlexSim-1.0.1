"""Mini banc de validation QoS clusters (S1–S3)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from loraflexsim.launcher.qos import QoSManager
from loraflexsim.scenarios import qos_cluster_bench as bench

from metrics import RunMetrics, compute_run_metrics
from plots import generate_plots


ALGORITHM_KEYS = ["adr", "apra", "mixrah", "mixraopt"]
ALGORITHM_LABELS = {
    "adr": "ADR",
    "apra": "APRA-like",
    "mixrah": "MixRA-H",
    "mixraopt": "MixRA-Opt",
}
TARGET_ALGOS = {"MixRA-H", "MixRA-Opt"}
BASELINE_ALGOS = {"ADR", "APRA-like"}
DEFAULT_DURATION_S = 4 * 3600.0
PDR_TOLERANCE = 0.02


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    num_nodes: int
    period_s: float


SCENARIOS: Sequence[ScenarioConfig] = (
    ScenarioConfig("S1", 1000, 600.0),
    ScenarioConfig("S2", 5000, 300.0),
    ScenarioConfig("S3", 10000, 150.0),
)


def _load_algorithms():
    specs = {spec.key: spec for spec in bench.ALGORITHMS if spec.key in ALGORITHM_KEYS}
    ordered = [specs[key] for key in ALGORITHM_KEYS if key in specs]
    if not ordered:
        raise RuntimeError("Aucun algorithme compatible trouvé dans le banc principal")
    return ordered


def _apply_algorithm(spec, simulator, manager: QoSManager, solver_mode: str) -> None:
    if spec.requires_qos:
        bench._configure_clusters(manager, simulator.packet_interval)
    spec.apply(simulator, manager, solver_mode)


def _run_single(
    scenario: ScenarioConfig,
    spec,
    seed: int,
    duration_s: float,
    solver_mode: str,
) -> RunMetrics:
    simulator = bench._create_simulator(scenario.num_nodes, scenario.period_s, seed)
    manager = QoSManager()
    _apply_algorithm(spec, simulator, manager, solver_mode)
    simulator.run(max_time=duration_s)
    base_metrics = simulator.get_metrics()
    base_metrics = dict(base_metrics)
    base_metrics["num_nodes"] = scenario.num_nodes
    base_metrics["packet_interval_s"] = scenario.period_s
    base_metrics["mixra_solver"] = getattr(simulator, "qos_mixra_solver", None)
    label = ALGORITHM_LABELS.get(spec.key, spec.label)
    result = compute_run_metrics(
        scenario=scenario.name,
        algorithm=label,
        base_metrics=base_metrics,
        events=getattr(simulator, "events_log", []),
    )
    return result


def _evaluate_targets(
    results: Mapping[Tuple[str, str], RunMetrics],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    table: Dict[str, Dict[str, str]] = {scenario.name: {} for scenario in SCENARIOS}
    notes: Dict[str, List[str]] = {scenario.name: [] for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for algorithm in ALGORITHM_LABELS.values():
            table[scenario.name][algorithm] = "PASS"
    for scenario in SCENARIOS:
        for algorithm in TARGET_ALGOS:
            entry = results.get((scenario.name, algorithm))
            if entry is None:
                table[scenario.name][algorithm] = "FAIL"
                notes[scenario.name].append(f"{algorithm}: aucun résultat")
                continue
            cluster_ids = sorted(entry.cluster_targets, key=lambda cid: int(cid))
            if scenario.name in {"S1", "S2"}:
                missing: List[str] = []
                for cid in cluster_ids:
                    actual = entry.cluster_pdr.get(cid, 0.0)
                    target = entry.cluster_targets.get(cid, 0.0)
                    if actual + 1e-9 < max(0.0, target - PDR_TOLERANCE):
                        missing.append(f"{cid} ({actual:.3f} < {target - PDR_TOLERANCE:.3f})")
                if missing:
                    table[scenario.name][algorithm] = "FAIL"
                    notes[scenario.name].append(
                        f"{algorithm}: clusters sous cible tolérée -> {', '.join(missing)}"
                    )
            else:
                below = []
                for cid in cluster_ids:
                    actual = entry.cluster_pdr.get(cid, 0.0)
                    target = entry.cluster_targets.get(cid, 0.0)
                    if actual + 1e-9 < target:
                        below.append(cid)
                if below:
                    first = below[0]
                    actual = entry.cluster_pdr.get(first, 0.0)
                    target = entry.cluster_targets.get(first, 0.0)
                    notes[scenario.name].append(
                        f"{algorithm}: rupture cluster {first} ({actual:.3f} < {target:.3f})"
                    )
                else:
                    table[scenario.name][algorithm] = "FAIL"
                    notes[scenario.name].append(
                        f"{algorithm}: aucune rupture détectée alors qu'attendue"
                    )
    scenario_s2 = results.get(("S2", "MixRA-H")), results.get(("S2", "MixRA-Opt"))
    target_pdr = [entry.pdr_global for entry in scenario_s2 if entry is not None]
    if target_pdr:
        best_target = min(target_pdr)
        for baseline in BASELINE_ALGOS:
            entry = results.get(("S2", baseline))
            if entry is None:
                table["S2"][baseline] = "FAIL"
                notes["S2"].append(f"{baseline}: aucun résultat")
                continue
            if entry.pdr_global >= best_target - 1e-9:
                table["S2"][baseline] = "FAIL"
                notes["S2"].append(
                    f"{baseline}: PDR global {entry.pdr_global:.3f} ≥ {best_target:.3f} (cible)"
                )
    return table, notes


def _write_summary(
    table: Mapping[str, Mapping[str, str]],
    notes: Mapping[str, Sequence[str]],
    output_dir: Path,
    solver_notes: Sequence[str],
) -> Path:
    lines: List[str] = []
    lines.append("Validation QoS clusters – résumé")
    lines.append("")
    header = ["Scénario"] + list(ALGORITHM_LABELS.values()) + ["Notes"]
    lines.append(" | ".join(header))
    lines.append(" | ".join("---" for _ in header))
    for scenario in SCENARIOS:
        status = [table.get(scenario.name, {}).get(algo, "-") for algo in ALGORITHM_LABELS.values()]
        comment = "; ".join(notes.get(scenario.name, []))
        lines.append(" | ".join([scenario.name] + status + [comment if comment else "-"]))
    if solver_notes:
        lines.append("")
        lines.append("Solveurs MixRA-Opt :")
        for note in solver_notes:
            lines.append(f"- {note}")
    summary_path = output_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(lines), encoding="utf8")
    return summary_path


def run_pipeline(
    output_dir: Path,
    *,
    seed: int = 1,
    duration_s: float = DEFAULT_DURATION_S,
    solver_mode: str = "auto",
) -> Dict[str, object]:
    specs = _load_algorithms()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[RunMetrics] = []
    solver_notes: List[str] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for algo_index, spec in enumerate(specs):
            combo_seed = seed + scenario_index * 10 + algo_index
            result = _run_single(scenario, spec, combo_seed, duration_s, solver_mode)
            results.append(result)
            if result.algorithm == "MixRA-Opt":
                solver_used = result.mixra_solver or solver_mode
                solver_notes.append(
                    f"{scenario.name}: MixRA-Opt -> {solver_used}"
                )
    mapping = {(item.scenario, item.algorithm): item for item in results}
    table, notes = _evaluate_targets(mapping)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].to_csv_row().keys()))
        writer.writeheader()
        for item in results:
            writer.writerow(item.to_csv_row())
    summary_path = _write_summary(table, notes, output_dir, solver_notes)
    figures = generate_plots(results, output_dir, scenario_order=[s.name for s in SCENARIOS])
    return {
        "results": results,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "figures": figures,
        "status_table": table,
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Répertoire de sortie")
    parser.add_argument("--seed", type=int, default=1, help="Graine initiale")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Durée maximale de simulation (secondes)",
    )
    parser.add_argument(
        "--solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Mode de solveur pour MixRA-Opt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_pipeline(
        Path(args.out),
        seed=args.seed,
        duration_s=args.duration,
        solver_mode=args.solver,
    )
    summary_path = report.get("summary_path")
    if summary_path:
        print(f"Résumé PASS/FAIL : {summary_path}")
    csv_path = report.get("csv_path")
    if csv_path:
        print(f"Métriques : {csv_path}")
    for figure in report.get("figures", []):
        print(f"Figure générée : {figure}")
    return report


if __name__ == "__main__":
    main()
