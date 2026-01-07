"""Trace des heatmaps DER/SNIR pour l'étape 1.

Le script génère des matrices (x = intervalle, y = nœuds) pour deux états
SNIR (on/off) empilés, avec une légende colorée adaptée aux publications
IEEE. Par défaut, les heatmaps sont filtrées sur l'algorithme mixra_opt.
"""

from __future__ import annotations

import argparse
import statistics
import sys
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

METRIC_LABELS = {
    "DER": ("DER", "DER"),
    "SNIR": ("snir_mean", "SNIR moyen (dB)"),
}

SNIR_STATES = ("snir_on", "snir_off")
SNIR_TITLES = {
    "snir_on": "SNIR activé",
    "snir_off": "SNIR désactivé",
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

    all_values = [
        value
        for _, _, matrix in matrices.values()
        for row in matrix
        for value in row
        if not (value != value)
    ]
    vmin = min(all_values) if all_values else None
    vmax = max(all_values) if all_values else None

    fig, axes = plt.subplots(
        len(SNIR_STATES),
        1,
        figsize=(8, 3.5 * len(SNIR_STATES)),
        sharex=True,
    )
    if len(SNIR_STATES) == 1:
        axes = [axes]

    image = None
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
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
        ax.set_ylabel("Nœuds")
        ax.set_title(f"{metric_label} – {SNIR_TITLES[state]}")
        ax.tick_params(axis="both", direction="in", length=4.5, width=1.0, labelsize=9)

    axes[-1].set_xlabel("Intervalle (s)")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, orientation="vertical", shrink=0.9, pad=0.02)
        cbar.set_label(metric_label)
        cbar.ax.tick_params(labelsize=9)

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"heatmap_{metric_key.lower()}_{algorithm.replace(' ', '_')}_snir_overlay.png"
    fig.tight_layout()
    fig.savefig(figures_dir / output_name, dpi=300)
    plt.close(fig)


def generate_heatmaps(
    results_dir: Path,
    figures_dir: Path,
    metric: str,
    algorithm: str | None,
    strict: bool,
    ieee_mode: bool,
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

    algorithms = {str(record.get("algorithm") or "") for record in records}
    if "mixra_opt" not in algorithms:
        warning = "Aucune donnée pour l'algorithme mixra_opt n'a été détectée."
        if ieee_mode:
            raise ValueError(f"{warning} (mode IEEE).")
        print(warning, file=sys.stderr)

    if algorithm:
        selected_algorithms = [algorithm]
    else:
        selected_algorithms = ["mixra_opt"]

    for algo in selected_algorithms:
        algo_records = [record for record in records if str(record.get("algorithm") or "unknown") == algo]
        if not algo_records:
            if ieee_mode:
                raise ValueError(
                    f"Aucune donnée pour l'algorithme {algo} (mode IEEE)."
                )
            continue
        _plot_heatmaps(algo_records, metric_key, metric_label, algo, figures_dir)


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
        "--algorithm",
        default="mixra_opt",
        help="Filtre sur l'algorithme à tracer (par défaut : mixra_opt)",
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
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    generate_heatmaps(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        metric=args.metric,
        algorithm=args.algorithm,
        strict=args.strict,
        ieee_mode=args.ieee,
    )


if __name__ == "__main__":
    main()
