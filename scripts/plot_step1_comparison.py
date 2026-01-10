"""Generate step 1 comparison figures from CSV outputs.

Figures are exported to figures/step1 as PNG and PDF:
- Figure 1: PDR/DER vs nodes, 3x2 grid (clusters x SNIR OFF/ON).
- Figure 2: Jain index vs nodes, 1x2 grid (SNIR OFF/ON).
- Figure 3: Throughput vs nodes, 1x2 grid (SNIR OFF/ON).
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency in CI
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - allow script import without matplotlib
    plt = None  # type: ignore

from plot_theme import SNIR_COLORS, apply_plot_theme

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step1"

STATE_LABELS = {True: "snir_on", False: "snir_off", None: "snir_unknown"}
SNIR_LABELS = {"snir_on": "SNIR ON", "snir_off": "SNIR OFF", "snir_unknown": "SNIR UNKNOWN"}

COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]
DEFAULT_ALGO_PRIORITY = ["adr", "apra", "mixra_h", "mixra_opt"]


def _normalize_algorithm_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mixra_opt": {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"},
        "mixra_h": {"mixra_h", "mixrah", "mixra-h", "mixra h"},
    }
    for canonical, names in aliases.items():
        if normalized in names:
            return canonical
    return text


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
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
        return None
    parsed = _parse_bool(row.get("use_snir") or row.get("with_snir"))
    return STATE_LABELS.get(parsed, "snir_unknown") if parsed is not None else None


def _load_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                algorithm = _normalize_algorithm_name(row.get("algorithm"))
                if not algorithm:
                    algorithm = _normalize_algorithm_name(csv_path.parent.name) or csv_path.parent.name
                snir_state = _snir_state_from_row(row)
                if snir_state is None:
                    warnings.warn(
                        f"No SNIR state found in {csv_path}; skipping row.",
                        RuntimeWarning,
                    )
                    continue
                cluster_pdr: Dict[int, float] = {}
                for key, value in row.items():
                    if key.startswith("qos_cluster_pdr__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_pdr[cluster_id] = _parse_float(value)
                record = {
                    "algorithm": algorithm,
                    "num_nodes": _parse_int(row.get("num_nodes")),
                    "snir_state": snir_state,
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "jain_index": _parse_float(row.get("jain_index")),
                    "throughput_bps": _parse_float(row.get("throughput_bps")),
                    "cluster_pdr": cluster_pdr,
                }
                records.append(record)
    return records


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean = fmean(values)
    std = pstdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, margin


def _collect_nodes(records: Iterable[Mapping[str, Any]]) -> List[int]:
    nodes = sorted({int(record["num_nodes"]) for record in records})
    return nodes


def _select_algorithms(records: Iterable[Mapping[str, Any]], selected: Sequence[str] | None) -> List[str]:
    if selected:
        return list(selected)
    available = {str(record["algorithm"]) for record in records if record.get("algorithm")}
    algorithms = [algo for algo in DEFAULT_ALGO_PRIORITY if algo in available]
    if not algorithms:
        return sorted(available)
    return algorithms


def _select_clusters(records: Iterable[Mapping[str, Any]], max_clusters: int = 3) -> List[int]:
    cluster_ids = sorted({cid for record in records for cid in record.get("cluster_pdr", {})})
    if len(cluster_ids) > max_clusters:
        warnings.warn(
            f"More than {max_clusters} clusters found; using the first {max_clusters}.",
            RuntimeWarning,
        )
        return cluster_ids[:max_clusters]
    return cluster_ids


def _style_map(labels: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for idx, label in enumerate(labels):
        mapping[label] = (COLOR_CYCLE[idx % len(COLOR_CYCLE)], MARKER_CYCLE[idx % len(MARKER_CYCLE)])
    return mapping


def _marker_map(labels: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for idx, label in enumerate(labels):
        mapping[label] = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
    return mapping


def _snir_color(state: str) -> str:
    return SNIR_COLORS.get(state, SNIR_COLORS["snir_unknown"])


def _plot_pdr_der(
    records: List[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
    clusters: Sequence[int],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures")
    states = ["snir_off", "snir_on"]
    n_rows = max(1, len(clusters))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]  # type: ignore[list-item]
    style = _style_map(list(algorithms))

    for row_idx, cluster_id in enumerate(clusters):
        for col_idx, state in enumerate(states):
            ax = axes[row_idx][col_idx]
            for algo in algorithms:
                algo_records = [
                    record
                    for record in records
                    if record.get("algorithm") == algo and record.get("snir_state") == state
                ]
                nodes = _collect_nodes(algo_records)
                pdr_means: List[float] = []
                pdr_cis: List[float] = []
                der_means: List[float] = []
                der_cis: List[float] = []
                for node in nodes:
                    node_records = [rec for rec in algo_records if rec.get("num_nodes") == node]
                    pdr_values = [
                        float(rec.get("cluster_pdr", {}).get(cluster_id, float("nan")))
                        for rec in node_records
                        if cluster_id in rec.get("cluster_pdr", {})
                    ]
                    der_values = [float(rec.get("DER", float("nan"))) for rec in node_records]
                    pdr_mean, pdr_ci = _mean_ci([val for val in pdr_values if not math.isnan(val)])
                    der_mean, der_ci = _mean_ci([val for val in der_values if not math.isnan(val)])
                    pdr_means.append(pdr_mean)
                    pdr_cis.append(pdr_ci)
                    der_means.append(der_mean)
                    der_cis.append(der_ci)
                color, marker = style[algo]
                ax.errorbar(
                    nodes,
                    pdr_means,
                    yerr=pdr_cis,
                    color=color,
                    marker=marker,
                    linestyle="-",
                    capsize=3,
                    label=f"{algo} PDR",
                )
                ax.errorbar(
                    nodes,
                    der_means,
                    yerr=der_cis,
                    color=color,
                    linestyle="--",
                    capsize=3,
                    label=f"{algo} DER",
                )
            ax.set_title(f"Cluster {cluster_id} - {SNIR_LABELS[state]}")
            ax.set_xlabel("Nodes")
            ax.set_ylabel("PDR / DER")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("PDR and DER vs Nodes")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"step1_pdr_der_comparison.{ext}")
    plt.close(fig)


def _plot_single_metric(
    records: List[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
    metric_key: str,
    title: str,
    y_label: str,
    filename: str,
    y_limits: Tuple[float, float] | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures")
    states = ["snir_off", "snir_on"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    style = _style_map(list(algorithms))

    for idx, state in enumerate(states):
        ax = axes[idx]
        for algo in algorithms:
            algo_records = [
                record
                for record in records
                if record.get("algorithm") == algo and record.get("snir_state") == state
            ]
            nodes = _collect_nodes(algo_records)
            means: List[float] = []
            cis: List[float] = []
            for node in nodes:
                node_records = [rec for rec in algo_records if rec.get("num_nodes") == node]
                values = [float(rec.get(metric_key, float("nan"))) for rec in node_records]
                mean, ci = _mean_ci([val for val in values if not math.isnan(val)])
                means.append(mean)
                cis.append(ci)
            color, marker = style[algo]
            ax.errorbar(
                nodes,
                means,
                yerr=cis,
                color=color,
                marker=marker,
                linestyle="-",
                capsize=3,
                label=algo,
            )
        ax.set_title(SNIR_LABELS[state])
        ax.set_xlabel("Nodes")
        ax.set_ylabel(y_label)
        if y_limits:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{ext}")
    plt.close(fig)


def _plot_pdr_der_overlay(
    records: List[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
    clusters: Sequence[int],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures")
    states = ["snir_off", "snir_on"]
    n_rows = max(1, len(clusters))
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]  # type: ignore[list-item]
    marker_map = _marker_map(list(algorithms))

    for row_idx, cluster_id in enumerate(clusters):
        ax = axes[row_idx]
        for algo in algorithms:
            algo_records = [record for record in records if record.get("algorithm") == algo]
            for state in states:
                state_records = [record for record in algo_records if record.get("snir_state") == state]
                nodes = _collect_nodes(state_records)
                pdr_means: List[float] = []
                pdr_cis: List[float] = []
                der_means: List[float] = []
                der_cis: List[float] = []
                for node in nodes:
                    node_records = [rec for rec in state_records if rec.get("num_nodes") == node]
                    pdr_values = [
                        float(rec.get("cluster_pdr", {}).get(cluster_id, float("nan")))
                        for rec in node_records
                        if cluster_id in rec.get("cluster_pdr", {})
                    ]
                    der_values = [float(rec.get("DER", float("nan"))) for rec in node_records]
                    pdr_mean, pdr_ci = _mean_ci([val for val in pdr_values if not math.isnan(val)])
                    der_mean, der_ci = _mean_ci([val for val in der_values if not math.isnan(val)])
                    pdr_means.append(pdr_mean)
                    pdr_cis.append(pdr_ci)
                    der_means.append(der_mean)
                    der_cis.append(der_ci)
                color = _snir_color(state)
                marker = marker_map[algo]
                ax.errorbar(
                    nodes,
                    pdr_means,
                    yerr=pdr_cis,
                    color=color,
                    marker=marker,
                    linestyle="-",
                    capsize=3,
                    label=f"{algo} {SNIR_LABELS[state]} PDR",
                )
                ax.errorbar(
                    nodes,
                    der_means,
                    yerr=der_cis,
                    color=color,
                    marker=marker,
                    linestyle="--",
                    capsize=3,
                    label=f"{algo} {SNIR_LABELS[state]} DER",
                )
        ax.set_title(f"Cluster {cluster_id} - SNIR ON/OFF")
        ax.set_xlabel("Nodes")
        ax.set_ylabel("PDR / DER")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("PDR and DER vs Nodes (SNIR ON/OFF overlay)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"step1_pdr_der_overlay.{ext}")
    plt.close(fig)


def _plot_single_metric_overlay(
    records: List[Mapping[str, Any]],
    output_dir: Path,
    algorithms: Sequence[str],
    metric_key: str,
    title: str,
    y_label: str,
    filename: str,
    y_limits: Tuple[float, float] | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures")
    states = ["snir_off", "snir_on"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    marker_map = _marker_map(list(algorithms))

    for algo in algorithms:
        algo_records = [record for record in records if record.get("algorithm") == algo]
        for state in states:
            state_records = [record for record in algo_records if record.get("snir_state") == state]
            nodes = _collect_nodes(state_records)
            means: List[float] = []
            cis: List[float] = []
            for node in nodes:
                node_records = [rec for rec in state_records if rec.get("num_nodes") == node]
                values = [float(rec.get(metric_key, float("nan"))) for rec in node_records]
                mean, ci = _mean_ci([val for val in values if not math.isnan(val)])
                means.append(mean)
                cis.append(ci)
            ax.errorbar(
                nodes,
                means,
                yerr=cis,
                color=_snir_color(state),
                marker=marker_map[algo],
                linestyle="-",
                capsize=3,
                label=f"{algo} {SNIR_LABELS[state]}",
            )
    ax.set_title(title)
    ax.set_xlabel("Nodes")
    ax.set_ylabel(y_label)
    if y_limits:
        ax.set_ylim(*y_limits)
    ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{ext}")
    plt.close(fig)


def generate_figures(
    results_dir: Path,
    output_dir: Path,
    algorithms: Sequence[str] | None = None,
    clusters: Sequence[int] | None = None,
    overlay_snir: bool = False,
    overlay_only: bool = False,
) -> None:
    if plt is not None:
        apply_plot_theme(plt)
    records = _load_records(results_dir)
    if not records:
        raise ValueError(f"No CSV records found in {results_dir}")
    selected_algorithms = _select_algorithms(records, algorithms)
    selected_clusters = list(clusters) if clusters else _select_clusters(records)
    if not selected_clusters:
        raise ValueError("No cluster PDR data found in results")

    if not overlay_only:
        _plot_pdr_der(records, output_dir, selected_algorithms, selected_clusters)
        _plot_single_metric(
            records,
            output_dir,
            selected_algorithms,
            "jain_index",
            "Jain Index vs Nodes",
            "Jain Index",
            "step1_jain_comparison",
            y_limits=(0.0, 1.0),
        )
        _plot_single_metric(
            records,
            output_dir,
            selected_algorithms,
            "throughput_bps",
            "Throughput vs Nodes",
            "Throughput (bps)",
            "step1_throughput_comparison",
            y_limits=None,
        )
    if overlay_snir or overlay_only:
        _plot_pdr_der_overlay(records, output_dir, selected_algorithms, selected_clusters)
        _plot_single_metric_overlay(
            records,
            output_dir,
            selected_algorithms,
            "jain_index",
            "Jain Index vs Nodes (SNIR ON/OFF overlay)",
            "Jain Index",
            "step1_jain_overlay",
            y_limits=(0.0, 1.0),
        )
        _plot_single_metric_overlay(
            records,
            output_dir,
            selected_algorithms,
            "throughput_bps",
            "Throughput vs Nodes (SNIR ON/OFF overlay)",
            "Throughput (bps)",
            "step1_throughput_overlay",
            y_limits=None,
        )


def _parse_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate step1 comparison figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing step1 CSV files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for figures (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated list of algorithms to plot (optional)",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default=None,
        help="Comma-separated list of cluster IDs to plot (optional)",
    )
    parser.add_argument(
        "--overlay-snir",
        action="store_true",
        help="Génère des figures superposées SNIR ON/OFF en complément.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Génère uniquement les figures superposées SNIR ON/OFF.",
    )
    args = parser.parse_args(argv)

    algorithms = _parse_list(args.algorithms)
    cluster_values = _parse_list(args.clusters)
    clusters = [int(value) for value in cluster_values] if cluster_values else None

    generate_figures(
        args.results_dir,
        args.output_dir,
        algorithms or None,
        clusters,
        overlay_snir=args.overlay_snir,
        overlay_only=args.overlay_only,
    )


if __name__ == "__main__":
    main()
