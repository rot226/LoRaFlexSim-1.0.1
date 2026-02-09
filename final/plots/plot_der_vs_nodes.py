"""Trace la DER moyenne vs taille de réseau (par cluster + global).

Ce script lit les CSV produits dans final/data/ et génère des figures dans
final/figures/. Il fonctionne hors ligne (aucune simulation).

Exemple (Windows 11, PowerShell) :
  python final/plots/plot_der_vs_nodes.py --data-dir final/data --output-dir final/figures
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:  # pragma: no cover - dépendance optionnelle
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - permet l'import sans matplotlib
    plt = None  # type: ignore

METHOD_ORDER = ["ADR", "APRA", "Aimi", "MixRA-H", "MixRA-Opt", "SNIR", "UCB1"]
COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]

ALGO_MAP = {
    "adr": "ADR",
    "apra": "APRA",
    "aimi": "Aimi",
    "mixra-h": "MixRA-H",
    "mixra_h": "MixRA-H",
    "mixra-opt": "MixRA-Opt",
    "mixra_opt": "MixRA-Opt",
}


def _parse_nodes(name: str) -> int | None:
    match = re.search(r"(?:N|nodes)[-_]?(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalize_algorithm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace(" ", "_")
    return ALGO_MAP.get(normalized, text)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _load_qos_like(
    base_dir: Path,
    scenario: str,
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Charge les CSV cluster_<id>.csv et renvoie cluster -> méthode -> nodes -> der."""
    results: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    scenario_dir = base_dir / scenario
    if not scenario_dir.exists():
        return results

    for node_dir in sorted(scenario_dir.iterdir()):
        if not node_dir.is_dir():
            continue
        nodes = _parse_nodes(node_dir.name)
        if nodes is None:
            continue
        for csv_path in sorted(node_dir.glob("cluster_*.csv")):
            cluster_match = re.search(r"cluster_(\d+)\.csv", csv_path.name)
            if not cluster_match:
                continue
            cluster_id = int(cluster_match.group(1))
            for row in _read_csv_rows(csv_path):
                algo = _normalize_algorithm(row.get("algorithme") or row.get("algorithm"))
                if not algo:
                    continue
                try:
                    der_mean = float(row.get("der_mean", "") or 0.0)
                except ValueError:
                    der_mean = 0.0
                results[cluster_id][algo][nodes] = der_mean
    return results


def _load_ucb1(base_dir: Path) -> Dict[int, Dict[str, Dict[int, float]]]:
    results: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    scenario_dir = base_dir / "ucb1"
    if not scenario_dir.exists():
        return results

    for node_dir in sorted(scenario_dir.iterdir()):
        if not node_dir.is_dir():
            continue
        nodes = _parse_nodes(node_dir.name)
        if nodes is None:
            continue
        for csv_path in sorted(node_dir.glob("cluster_*.csv")):
            cluster_match = re.search(r"cluster_(\d+)\.csv", csv_path.name)
            if not cluster_match:
                continue
            cluster_id = int(cluster_match.group(1))
            rows = _read_csv_rows(csv_path)
            if not rows:
                continue
            row = rows[0]
            try:
                der_mean = float(row.get("der_mean", "") or 0.0)
            except ValueError:
                der_mean = 0.0
            results[cluster_id]["UCB1"][nodes] = der_mean
    return results


def _select_snir_method(
    snir_results: Dict[int, Dict[str, Dict[int, float]]],
    preferred_algo: str = "ADR",
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Expose un seul profil SNIR (par défaut ADR) sous la clé SNIR."""
    selected: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for cluster_id, algo_map in snir_results.items():
        if preferred_algo in algo_map:
            selected[cluster_id]["SNIR"] = algo_map[preferred_algo]
        elif algo_map:
            algo_name = sorted(algo_map.keys())[0]
            selected[cluster_id]["SNIR"] = algo_map[algo_name]
            warnings.warn(
                f"Algorithme SNIR préféré '{preferred_algo}' absent pour cluster {cluster_id}; "
                f"utilise '{algo_name}'.",
                RuntimeWarning,
            )
    return selected


def _merge_cluster_metrics(
    *entries: Dict[int, Dict[str, Dict[int, float]]]
) -> Dict[int, Dict[str, Dict[int, float]]]:
    merged: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for entry in entries:
        for cluster_id, method_map in entry.items():
            for method, values in method_map.items():
                merged[cluster_id][method].update(values)
    return merged


def _build_global(
    cluster_data: Dict[int, Dict[str, Dict[int, float]]],
) -> Dict[str, Dict[int, float]]:
    global_data: Dict[str, Dict[int, float]] = defaultdict(dict)
    for cluster_id, method_map in cluster_data.items():
        for method, values in method_map.items():
            for nodes, value in values.items():
                global_data.setdefault(method, {}).setdefault(nodes, [])
                if isinstance(global_data[method][nodes], list):
                    global_data[method][nodes].append(value)
    for method, nodes_map in list(global_data.items()):
        for nodes, values in list(nodes_map.items()):
            if isinstance(values, list) and values:
                global_data[method][nodes] = sum(values) / len(values)
            else:
                global_data[method][nodes] = math.nan
    return global_data


def _plot_lines(
    ax: Any,
    nodes: Iterable[int],
    series: Mapping[str, Mapping[int, float]],
) -> None:
    style_map = {
        method: (COLOR_CYCLE[idx % len(COLOR_CYCLE)], MARKER_CYCLE[idx % len(MARKER_CYCLE)])
        for idx, method in enumerate(METHOD_ORDER)
    }
    for method in METHOD_ORDER:
        values = series.get(method, {})
        if not values:
            continue
        xs = [n for n in nodes if n in values]
        ys = [values[n] for n in xs]
        if not xs:
            continue
        color, marker = style_map[method]
        ax.plot(xs, ys, label=method, color=color, marker=marker, linewidth=1.6)


def _collect_nodes(cluster_data: Dict[int, Dict[str, Dict[int, float]]]) -> List[int]:
    nodes = {n for method_map in cluster_data.values() for values in method_map.values() for n in values}
    return sorted(nodes)


def plot_der_vs_nodes(data_dir: Path, output_dir: Path) -> Path:
    qos_results = _load_qos_like(data_dir, "qos_baselines")
    snir_results = _load_qos_like(data_dir, "snir")
    snir_selected = _select_snir_method(snir_results)
    ucb1_results = _load_ucb1(data_dir)
    cluster_data = _merge_cluster_metrics(qos_results, snir_selected, ucb1_results)

    if not cluster_data:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {data_dir}")

    nodes = _collect_nodes(cluster_data)
    cluster_ids = sorted(cluster_data.keys())
    global_data = _build_global(cluster_data)

    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib est requis pour générer les figures.")

    total_plots = len(cluster_ids) + 1
    cols = 2
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes_flat[idx]
        ax.set_title(f"Cluster {cluster_id}")
        _plot_lines(ax, nodes, cluster_data[cluster_id])
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel("DER moyenne")
        ax.grid(True, alpha=0.3)

    global_ax = axes_flat[len(cluster_ids)]
    global_ax.set_title("Global (moyenne des clusters)")
    _plot_lines(global_ax, nodes, global_data)
    global_ax.set_xlabel("Nombre de nœuds")
    global_ax.set_ylabel("DER moyenne")
    global_ax.grid(True, alpha=0.3)

    for ax in axes_flat[total_plots:]:
        ax.axis("off")

    handles, labels = global_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "der_vs_nodes.png"
    pdf_path = output_dir / "der_vs_nodes.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace la DER moyenne par cluster vs nombre de nœuds.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final/data"),
        help="Dossier racine contenant les CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final/figures"),
        help="Dossier où écrire les figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_der_vs_nodes(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
