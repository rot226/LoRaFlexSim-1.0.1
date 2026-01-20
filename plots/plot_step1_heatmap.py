"""Trace des heatmaps DER/SNIR pour l'étape 1.

Le script génère des matrices (x = intervalle, y = nœuds) pour deux états
SNIR (on/off) empilés, avec une légende claire et une palette uniforme. Par
défaut, les heatmaps sont produites pour les 4 algorithmes (adr, apra,
mixra_h, mixra_opt).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - dépend de l'environnement d'exécution
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.plot_step1_results import (  # noqa: E402
    _apply_ieee_style,
    _load_step1_records,
    _select_signal_mean,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step1" / "heatmaps"
DEFAULT_ALGORITHMS = ["adr", "apra", "mixra_h", "mixra_opt"]
HEATMAP_COLORMAP = "cividis"

METRIC_LABELS = {
    "DER": ("DER", "DER (probability)"),
    "SNIR": ("snir_mean", "Mean SNIR (dB)"),
}

SNIR_STATES = ("snir_on", "snir_off")
SNIR_TITLES = {
    "snir_on": "SNIR enabled",
    "snir_off": "SNIR disabled",
}


def _metric_value(record: Mapping[str, object], metric_key: str) -> float | None:
    if metric_key == "snir_mean":
        value, _error_metric = _select_signal_mean(record)
    else:
        value = record.get(metric_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_values(records: Iterable[Mapping[str, object]], metric_key: str) -> float | None:
    values: List[float] = []
    for record in records:
        value = _metric_value(record, metric_key)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return statistics.mean(values)


def _build_matrix(
    records: Sequence[Mapping[str, float | int | str]],
    metric_key: str,
) -> Tuple[List[int], List[float], List[List[float]]]:
    nodes = sorted({int(record["num_nodes"]) for record in records})
    intervals = sorted({float(record["packet_interval_s"]) for record in records})

    matrix: List[List[float]] = []
    for node in nodes:
        row: List[float] = []
        for interval in intervals:
            cell_records = [
                record
                for record in records
                if int(record["num_nodes"]) == node
                and float(record["packet_interval_s"]) == interval
            ]
            aggregated = _aggregate_values(cell_records, metric_key)
            row.append(float("nan") if aggregated is None else aggregated)
        matrix.append(row)
    return nodes, intervals, matrix


def _format_axis_labels(values: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for value in values:
        if float(value).is_integer():
            labels.append(f"{int(value)}")
        else:
            labels.append(f"{value:g}")
    return labels


def _plot_heatmaps(
    records: Sequence[Mapping[str, float | int | str]],
    metric_key: str,
    metric_label: str,
    algorithm: str,
    figures_dir: Path,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if plt is None:
        return

    by_state: Dict[str, List[Mapping[str, float | int | str]]] = {
        state: [r for r in records if r.get("snir_state") == state]
        for state in SNIR_STATES
    }

    matrices: Dict[str, Tuple[List[int], List[float], List[List[float]]]] = {}
    for state, state_records in by_state.items():
        if not state_records:
            continue
        matrices[state] = _build_matrix(state_records, metric_key)

    if not matrices:
        return

    fig, axes = plt.subplots(
        len(SNIR_STATES),
        1,
        figsize=(8, 3.5 * len(SNIR_STATES)),
        sharex=True,
    )
    if len(SNIR_STATES) == 1:
        axes = [axes]

    image = None
    cmap = plt.get_cmap(HEATMAP_COLORMAP).copy()
    missing_color = "#f2f2f2"
    cmap.set_bad(color=missing_color)
    for ax, state in zip(axes, SNIR_STATES):
        if state not in matrices:
            ax.axis("off")
            continue
        nodes, intervals, matrix = matrices[state]
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels([str(node) for node in nodes])
        ax.set_xticks(range(len(intervals)))
        ax.set_xticklabels(_format_axis_labels(intervals))
        ax.set_ylabel("Network size (number of nodes)")
        ax.set_title(f"{metric_label} – {SNIR_TITLES[state]}")
        ax.tick_params(axis="both", direction="in", length=4.5, width=1.0, labelsize=9)

    axes[-1].set_xlabel("Interval (s)")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, orientation="vertical", shrink=0.9, pad=0.02)
        cbar.set_label(f"{metric_label} (palette uniforme)")
        cbar.ax.tick_params(labelsize=9)

    missing_patch = plt.matplotlib.patches.Patch(
        facecolor=missing_color,
        edgecolor="#666666",
        label="Données manquantes",
    )
    fig.legend(handles=[missing_patch], loc="lower center", ncol=1, frameon=False)
    fig.suptitle(f"Heatmap {metric_label} – {algorithm}", y=1.02, fontsize=11)

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"heatmap_{metric_key.lower()}_{algorithm.replace(' ', '_')}_snir_on_off.png"
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(figures_dir / output_name, dpi=300)
    plt.close(fig)


def _parse_algorithms(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    algorithms: List[str] = []
    for value in values:
        algorithms.extend([item.strip() for item in value.split(",") if item.strip()])
    return algorithms


def _metric_extent(
    records: Sequence[Mapping[str, float | int | str]],
    metric_key: str,
) -> tuple[float | None, float | None]:
    values: List[float] = []
    for record in records:
        value = _metric_value(record, metric_key)
        if value is None:
            continue
        values.append(value)
    if not values:
        return None, None
    return min(values), max(values)


def _filter_network_sizes(
    records: Sequence[Mapping[str, float | int | str]],
    network_sizes: Sequence[int] | None,
) -> list[Mapping[str, float | int | str]]:
    if not network_sizes:
        return list(records)
    available = sorted(
        {
            int(record["num_nodes"])
            for record in records
            if record.get("num_nodes") is not None
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
        record
        for record in records
        if int(record.get("num_nodes", -1)) in requested
    ]


def generate_heatmaps(
    results_dir: Path,
    figures_dir: Path,
    metric: str,
    algorithms: Sequence[str] | None,
    strict: bool,
    ieee_mode: bool,
    network_sizes: Sequence[int] | None,
) -> None:
    if plt is None:
        print("matplotlib n'est pas disponible ; aucune heatmap générée.")
        return

    metric_key, metric_label = METRIC_LABELS[metric]
    _apply_ieee_style()

    records = _load_step1_records(results_dir, strict=strict)
    if not records:
        if ieee_mode:
            raise FileNotFoundError(
                f"Aucun CSV trouvé dans {results_dir} (mode IEEE)."
            )
        print(f"Aucun CSV trouvé dans {results_dir} ; aucune heatmap générée.")
        return

    records = _filter_network_sizes(records, network_sizes)
    available_algorithms = {str(record.get("algorithm") or "") for record in records}

    requested_algorithms = _parse_algorithms(algorithms)
    if requested_algorithms:
        selected_algorithms = requested_algorithms
    else:
        selected_algorithms = DEFAULT_ALGORITHMS

    missing_algorithms = [algo for algo in selected_algorithms if algo not in available_algorithms]
    if missing_algorithms:
        warning = (
            "Aucune donnée détectée pour : "
            f"{', '.join(missing_algorithms)}."
        )
        if ieee_mode:
            raise ValueError(f"{warning} (mode IEEE).")
        print(warning, file=sys.stderr)

    filtered_records = [
        record
        for record in records
        if str(record.get("algorithm") or "") in selected_algorithms
    ]
    vmin, vmax = _metric_extent(filtered_records, metric_key)

    for algo in selected_algorithms:
        algo_records = [record for record in records if str(record.get("algorithm") or "unknown") == algo]
        if not algo_records:
            if ieee_mode:
                raise ValueError(
                    f"Aucune donnée pour l'algorithme {algo} (mode IEEE)."
                )
            continue
        _plot_heatmaps(
            algo_records,
            metric_key,
            metric_label,
            algo,
            figures_dir,
            vmin,
            vmax,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant les CSV de l'étape 1",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire de sortie des heatmaps",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_LABELS.keys()),
        default="DER",
        help="Métrique à représenter dans la heatmap (DER ou SNIR)",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help=(
            "Liste d'algorithmes séparés par des virgules (par défaut : "
            "adr, apra, mixra_h, mixra_opt)"
        ),
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Alias de --algorithms pour un seul algorithme.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Applique un filtrage strict des CSV (snir_state, snir_mean, snir_histogram_json) "
            "comme pour les figures extended."
        ),
    )
    parser.add_argument(
        "--ieee",
        action="store_true",
        help="Échoue si les CSV requis sont absents (mode IEEE).",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    generate_heatmaps(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        metric=args.metric,
        algorithms=(
            [args.algorithm]
            if args.algorithm and not args.algorithms
            else args.algorithms
        ),
        strict=args.strict,
        ieee_mode=args.ieee,
        network_sizes=args.network_sizes,
    )


if __name__ == "__main__":
    main()
