from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mne3sd.common import apply_ieee_style

DEFAULT_INPUT = Path("data/der_density.csv")
DEFAULT_OUTPUT_DIR = Path("plots")
ALGORITHM_ORDER = ["baseline", "baseline+SNIR"]
ALGORITHM_COLORS = {"baseline": "#1f77b4", "baseline+SNIR": "#ff7f0e"}


def _parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _load_records(csv_path: Path) -> List[Mapping[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    with csv_path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _filter_network_sizes(
    records: List[Mapping[str, str]],
    network_sizes: Sequence[int] | None,
) -> List[Mapping[str, str]]:
    if not network_sizes:
        return records
    available = sorted(
        {
            int(float(row.get("num_nodes", "0") or 0))
            for row in records
            if row.get("num_nodes") is not None
        }
    )
    requested = sorted({int(size) for size in network_sizes})
    missing = sorted(set(requested) - set(available))
    if missing:
        warnings.warn(
            "Tailles de réseau demandées absentes: "
            + ", ".join(str(size) for size in missing),
            stacklevel=2,
        )
    return [
        row
        for row in records
        if int(float(row.get("num_nodes", "0") or 0)) in requested
    ]


def _cluster_sort_value(cluster_label: str) -> Tuple[int, str]:
    stripped = cluster_label.strip().rstrip("%")
    try:
        return int(float(stripped)), cluster_label
    except ValueError:
        return 0, cluster_label


def _group_by_cluster(records: Iterable[Mapping[str, str]]) -> Dict[str, Dict[str, List[Tuple[int, float, float]]]]:
    grouped: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        try:
            num_nodes = int(float(row.get("num_nodes", "0") or 0))
        except ValueError:
            continue
        cluster = str(row.get("cluster", "")).strip() or "inconnu"
        algorithm = str(row.get("algorithm", "")).strip() or "inconnu"
        pdr = _parse_float(row.get("pdr"))
        der = _parse_float(row.get("der"))
        grouped[cluster][algorithm].append((num_nodes, pdr, der))

    for cluster_data in grouped.values():
        for algo_entries in cluster_data.values():
            algo_entries.sort(key=lambda item: item[0])
    return grouped


def _build_axes(num_clusters: int, *, double_column: bool):
    width_multiplier = 3.6 * (2 if double_column else 1)
    fig, axes = plt.subplots(
        2,
        num_clusters,
        figsize=(width_multiplier * num_clusters, 6.2),
        sharex="col",
    )
    if num_clusters == 1:
        axes = [[axes[0]], [axes[1]]]  # type: ignore[list-item]
    return fig, axes


def _plot_for_cluster(
    axes: Sequence,
    cluster_label: str,
    cluster_data: Mapping[str, Sequence[Tuple[int, float, float]]],
    pdr_target: float,
    show_target_label: bool,
) -> bool:
    plotted = False
    pdr_ax, der_ax = axes
    for algorithm in ALGORITHM_ORDER:
        entries = cluster_data.get(algorithm, [])
        if not entries:
            continue
        xs = [item[0] for item in entries]
        pdrs = [item[1] for item in entries]
        ders = [item[2] for item in entries]
        color = ALGORITHM_COLORS.get(algorithm)
        pdr_ax.plot(xs, pdrs, marker="o", label=algorithm, color=color)
        der_ax.plot(xs, ders, marker="o", label=algorithm, color=color)
        plotted = True

    pdr_ax.axhline(
        pdr_target,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label="PDR cible" if show_target_label else None,
    )
    pdr_ax.set_title(f"Cluster {cluster_label}")
    pdr_ax.set_ylabel("PDR (probability)")
    pdr_ax.set_ylim(0.0, 1.05)
    pdr_ax.grid(True, linestyle=":", alpha=0.5)

    der_ax.set_xlabel("Nombre de nœuds")
    der_ax.set_ylabel("DER (probability)")
    der_ax.set_ylim(0.0, 1.05)
    der_ax.grid(True, linestyle=":", alpha=0.5)
    return plotted


def plot_der_density(
    csv_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pdr_target: float = 0.9,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    double_column: bool = False,
    ieee_style: bool = False,
    network_sizes: Sequence[int] | None = None,
) -> List[Path]:
    formats = [ext.lower().lstrip(".") for ext in formats if ext]
    if not formats:
        raise ValueError("Aucun format de sortie fourni.")

    if ieee_style:
        apply_ieee_style(figsize=(7.2 if double_column else 3.6, 6.2))
    records = _filter_network_sizes(_load_records(csv_path), network_sizes)
    grouped = _group_by_cluster(records)
    if not grouped:
        raise ValueError("Aucune donnée à tracer.")

    clusters = sorted(grouped.keys(), key=_cluster_sort_value)
    fig, axes = _build_axes(len(clusters), double_column=double_column)
    plotted = False

    for idx, cluster_label in enumerate(clusters):
        cluster_axes = (axes[0][idx], axes[1][idx])
        plotted |= _plot_for_cluster(cluster_axes, cluster_label, grouped[cluster_label], pdr_target, idx == 0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
        )
    fig.suptitle("PDR/DER en fonction de la densité – comparaison baseline vs baseline+SNIR")
    plt.subplots_adjust(top=0.80)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for fmt in dict.fromkeys(formats):
        suffix = f".{fmt}"
        output_path = output_dir / f"der_density{suffix}"
        fig.savefig(output_path, dpi=150, format=fmt)
        saved_paths.append(output_path)
    plt.close(fig)

    if not plotted:
        raise ValueError("Aucune courbe tracée : vérifiez les algorithmes disponibles dans le CSV.")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace PDR/DER vs nombre de nœuds pour chaque cluster.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Chemin du CSV der_density.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dossier de sortie des figures.")
    parser.add_argument(
        "--pdr-target",
        type=float,
        default=0.9,
        help="Valeur cible de PDR pour la ligne horizontale.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Formats de sortie séparés par des virgules (ex: png,pdf,eps).",
    )
    parser.add_argument(
        "--double-column",
        action="store_true",
        help="Utiliser une largeur de figure double colonne.",
    )
    parser.add_argument(
        "--ieee-style",
        action="store_true",
        help="Appliquer le style rcParams compact pour publication IEEE.",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [item.strip() for part in args.formats.split(",") for item in [part] if item.strip()]
    saved_paths = plot_der_density(
        args.input,
        args.output_dir,
        args.pdr_target,
        formats=formats,
        double_column=args.double_column,
        ieee_style=args.ieee_style,
        network_sizes=args.network_sizes,
    )
    print("Figures enregistrées dans :", ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
