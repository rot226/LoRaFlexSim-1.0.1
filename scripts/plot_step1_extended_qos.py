"""Génère des figures QoS étendues pour l'étape 1 (histogrammes, ECDF, heatmaps)."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

from scripts.plot_step1_results import _detect_snir_state, _normalize_algorithm_name

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step1" / "extended"

SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4"}
METRICS = ("PDR", "DER")
CLUSTER_PATTERN = re.compile(r"cluster_(pdr|der)[^0-9]*([0-9]+)", re.IGNORECASE)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                algorithm = _normalize_algorithm_name(row.get("algorithm"))
                if not algorithm:
                    algorithm = _normalize_algorithm_name(csv_path.parent.name) or csv_path.parent.name
                snir_state, snir_detected = _detect_snir_state(row)
                if not snir_detected or snir_state not in {"snir_on", "snir_off"}:
                    continue
                cluster_values: Dict[str, Dict[int, float]] = {"PDR": {}, "DER": {}}
                for key, raw_value in row.items():
                    match = CLUSTER_PATTERN.match(str(key))
                    if not match:
                        continue
                    metric_key = match.group(1).upper()
                    cluster_id = int(match.group(2))
                    value = _parse_float(raw_value)
                    if value is not None:
                        cluster_values[metric_key][cluster_id] = value
                record = {
                    "algorithm": algorithm,
                    "snir_state": snir_state,
                    "num_nodes": _parse_float(row.get("num_nodes")),
                    "packet_interval_s": _parse_float(row.get("packet_interval_s")),
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "cluster_values": cluster_values,
                }
                records.append(record)
    return records


def _group_by_algorithm(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("algorithm") or "unknown")].append(dict(record))
    return grouped


def _plot_histograms(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    by_algorithm = _group_by_algorithm(records)
    for algorithm, items in by_algorithm.items():
        cluster_ids: set[int] = set()
        for item in items:
            for metric in METRICS:
                cluster_ids.update(item.get("cluster_values", {}).get(metric, {}).keys())
        for cluster_id in sorted(cluster_ids):
            for metric in METRICS:
                values_by_state: Dict[str, List[float]] = {"snir_on": [], "snir_off": []}
                for item in items:
                    state = item.get("snir_state")
                    cluster_values = item.get("cluster_values", {}).get(metric, {})
                    value = cluster_values.get(cluster_id)
                    if state in values_by_state and value is not None:
                        values_by_state[state].append(float(value))
                if not any(values_by_state.values()):
                    continue
                fig, ax = plt.subplots(figsize=(6.6, 4.2))
                bins = 15
                for state, values in values_by_state.items():
                    if not values:
                        continue
                    ax.hist(
                        values,
                        bins=bins,
                        alpha=0.55,
                        color=SNIR_COLORS[state],
                        label=SNIR_LABELS[state],
                        edgecolor="white",
                    )
                ax.set_title(f"{metric} cluster {cluster_id} ({algorithm})")
                ax.set_xlabel(metric)
                ax.set_ylabel("Occurrences")
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.5)
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"step1_hist_{metric.lower()}_cluster{cluster_id}_{algorithm}"
                fig.tight_layout()
                fig.savefig(output_dir / f"{stem}.png", dpi=200)
                fig.savefig(output_dir / f"{stem}.pdf")
                plt.close(fig)


def _compute_ecdf(values: Sequence[float]) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    xs = sorted_vals
    ys = [(idx + 1) / n for idx in range(n)]
    return xs, ys


def _plot_ecdf(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    by_algorithm = _group_by_algorithm(records)
    for algorithm, items in by_algorithm.items():
        for metric in METRICS:
            values_by_state: Dict[str, List[float]] = {"snir_on": [], "snir_off": []}
            for item in items:
                state = item.get("snir_state")
                value = item.get(metric)
                if state in values_by_state and value is not None:
                    values_by_state[state].append(float(value))
            if not any(values_by_state.values()):
                continue
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            for state, values in values_by_state.items():
                if not values:
                    continue
                xs, ys = _compute_ecdf(values)
                ax.step(xs, ys, where="post", color=SNIR_COLORS[state], label=SNIR_LABELS[state], linewidth=2)
            ax.set_title(f"ECDF {metric} ({algorithm})")
            ax.set_xlabel(metric)
            ax.set_ylabel("ECDF")
            ax.set_ylim(0.0, 1.05)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"step1_ecdf_{metric.lower()}_{algorithm}"
            fig.tight_layout()
            fig.savefig(output_dir / f"{stem}.png", dpi=200)
            fig.savefig(output_dir / f"{stem}.pdf")
            plt.close(fig)


def _unique_sorted(values: Iterable[float]) -> List[float]:
    return sorted({value for value in values if value is not None and not math.isnan(value)})


def _plot_heatmaps(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    by_algorithm = _group_by_algorithm(records)
    for algorithm, items in by_algorithm.items():
        for metric in METRICS:
            for state in ("snir_on", "snir_off"):
                filtered = [item for item in items if item.get("snir_state") == state]
                if not filtered:
                    continue
                nodes = _unique_sorted(item.get("num_nodes") for item in filtered)
                intervals = _unique_sorted(item.get("packet_interval_s") for item in filtered)
                if not nodes or not intervals:
                    continue
                grid = [[math.nan for _ in intervals] for _ in nodes]
                for item in filtered:
                    node = item.get("num_nodes")
                    interval = item.get("packet_interval_s")
                    value = item.get(metric)
                    if node is None or interval is None or value is None:
                        continue
                    try:
                        row_idx = nodes.index(float(node))
                        col_idx = intervals.index(float(interval))
                    except ValueError:
                        continue
                    grid[row_idx][col_idx] = float(value)
                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"Heatmap {metric} ({algorithm}, {SNIR_LABELS[state]})")
                ax.set_xlabel("Intervalle paquet (s)")
                ax.set_ylabel("Nombre de nœuds")
                ax.set_xticks(range(len(intervals)), [str(val) for val in intervals])
                ax.set_yticks(range(len(nodes)), [str(val) for val in nodes])
                fig.colorbar(image, ax=ax, label=metric)
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"step1_heatmap_{metric.lower()}_{algorithm}_{state}"
                fig.tight_layout()
                fig.savefig(output_dir / f"{stem}.png", dpi=200)
                fig.savefig(output_dir / f"{stem}.pdf")
                plt.close(fig)


def generate_extended_qos_figures(results_dir: Path, output_dir: Path) -> None:
    records = _load_records(results_dir)
    if not records:
        print(f"Aucune donnée trouvée dans {results_dir}.")
        return
    _plot_histograms(records, output_dir)
    _plot_ecdf(records, output_dir)
    _plot_heatmaps(records, output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant les CSV Step1/QoS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire de sortie pour les figures (PNG/PDF).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    generate_extended_qos_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
