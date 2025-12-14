from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

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


def _build_axes(num_clusters: int):
    fig, axes = plt.subplots(2, num_clusters, figsize=(4.5 * num_clusters, 6.2), sharex="col")
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
    pdr_ax.set_ylabel("PDR")
    pdr_ax.set_ylim(0.0, 1.05)
    pdr_ax.grid(True, linestyle=":", alpha=0.5)

    der_ax.set_xlabel("Nombre de nœuds")
    der_ax.set_ylabel("DER")
    der_ax.set_ylim(0.0, 1.05)
    der_ax.grid(True, linestyle=":", alpha=0.5)
    return plotted


def plot_der_density(
    csv_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pdr_target: float = 0.9,
) -> Tuple[Path, Path]:
    records = _load_records(csv_path)
    grouped = _group_by_cluster(records)
    if not grouped:
        raise ValueError("Aucune donnée à tracer.")

    clusters = sorted(grouped.keys(), key=_cluster_sort_value)
    fig, axes = _build_axes(len(clusters))
    plotted = False

    for idx, cluster_label in enumerate(clusters):
        cluster_axes = (axes[0][idx], axes[1][idx])
        plotted |= _plot_for_cluster(cluster_axes, cluster_label, grouped[cluster_label], pdr_target, idx == 0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.suptitle("PDR/DER en fonction de la densité – comparaison baseline vs baseline+SNIR")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "der_density.png"
    pdf_path = output_dir / "der_density.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    if not plotted:
        raise ValueError("Aucune courbe tracée : vérifiez les algorithmes disponibles dans le CSV.")

    return png_path, pdf_path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, pdf_path = plot_der_density(args.input, args.output_dir, args.pdr_target)
    print(f"Figures enregistrées dans : {png_path} et {pdf_path}")


if __name__ == "__main__":
    main()
