"""Trace des boxplots/violons pour les distributions des métriques de l'étape 1.

Le script génère une figure par métrique et par algorithme, comparant les
états SNIR (on/off) avec un boxplot et un violon côte à côte. Chaque algorithme
est traité une seule fois afin d'éviter les duplications.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

try:  # pragma: no cover - dépend de l'environnement d'exécution
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.plot_step1_results import (  # noqa: E402
    _apply_ieee_style,
    _load_step1_records,
    _record_matches_state,
    _snir_color,
    _snir_label,
    _format_axes,
    _select_signal_mean,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step1" / "distributions"

SNIR_STATES = ("snir_on", "snir_off")

METRICS = {
    "PDR": "Overall PDR (probability)",
    "DER": "Overall DER (probability)",
    "snir_mean": "Mean SNIR (dB)",
    "snr_mean": "Mean SNR (dB)",
    "collisions": "Collisions (probability)",
    "collisions_snir": "Collisions (SNIR, probability)",
    "jain_index": "Jain index (unitless)",
    "throughput_bps": "Aggregate throughput (bps)",
}


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(record: Mapping[str, object], metric: str) -> float | None:
    if metric == "snir_mean":
        value, _error_metric = _select_signal_mean(record)
        return _as_float(value)
    raw = record.get(metric)
    if raw is None:
        raw = record.get(f"{metric}_mean")
    return _as_float(raw)


def _collect_values(
    records: Iterable[Mapping[str, object]],
    metric: str,
    state: str,
) -> List[float]:
    values: List[float] = []
    for record in records:
        if not _record_matches_state(record, state):
            continue
        value = _metric_value(record, metric)
        if value is None:
            continue
        values.append(value)
    return values


def _plot_distribution(
    records: Sequence[Mapping[str, object]],
    metric: str,
    label: str,
    algorithm: str,
    figures_dir: Path,
) -> None:
    if plt is None:
        return

    grouped = {state: _collect_values(records, metric, state) for state in SNIR_STATES}
    if not any(grouped.values()):
        return

    fig, axes = plt.subplots(2, len(SNIR_STATES), figsize=(10.0, 6.0), sharey="row")

    for col, state in enumerate(SNIR_STATES):
        values = grouped[state]
        box_ax = axes[0][col]
        violin_ax = axes[1][col]

        if values:
            boxplot = box_ax.boxplot(
                [values],
                patch_artist=True,
                medianprops={"color": "#000000", "linewidth": 1.3},
                boxprops={"linewidth": 1.2},
                whiskerprops={"linewidth": 1.1},
                capprops={"linewidth": 1.1},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(_snir_color(state))
                patch.set_alpha(0.5)
        else:
            box_ax.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=box_ax.transAxes,
            )
            box_ax.set_xticks([])

        if values:
            violin = violin_ax.violinplot(
                [values],
                showmeans=True,
                showmedians=False,
                showextrema=True,
            )
            for body in violin.get("bodies", []):
                body.set_facecolor(_snir_color(state))
                body.set_edgecolor("#333333")
                body.set_alpha(0.6)
        else:
            violin_ax.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=violin_ax.transAxes,
            )
            violin_ax.set_xticks([])

        box_ax.set_title(_snir_label(state))
        for ax in (box_ax, violin_ax):
            ax.set_xticks([])
            _format_axes(ax, integer_x=False)

        if col == 0:
            box_ax.set_ylabel(f"{label}\n(Box plot)")
            violin_ax.set_ylabel(f"{label}\n(Violin)")

    fig.suptitle(f"{label} – {algorithm} – distributions SNIR séparées")
    figures_dir.mkdir(parents=True, exist_ok=True)
    safe_algorithm = algorithm.replace(" ", "_")
    output = figures_dir / f"step1_distribution_{metric}_{safe_algorithm}.png"
    plt.subplots_adjust(top=0.80)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _unique_algorithms(algorithms: Iterable[str]) -> tuple[List[str], List[str]]:
    unique: List[str] = []
    seen: set[str] = set()
    duplicates: List[str] = []
    for algorithm in algorithms:
        if algorithm in seen:
            duplicates.append(algorithm)
            continue
        seen.add(algorithm)
        unique.append(algorithm)
    return unique, duplicates


def _parse_algorithms(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    algorithms: List[str] = []
    for value in values:
        algorithms.extend([item.strip() for item in value.split(",") if item.strip()])
    return algorithms


def generate_distributions(
    results_dir: Path,
    figures_dir: Path,
    algorithms: Sequence[str] | None,
    strict: bool,
    ieee_mode: bool,
) -> None:
    if plt is None:
        print("matplotlib n'est pas disponible ; aucune distribution générée.")
        return

    _apply_ieee_style()
    records = _load_step1_records(results_dir, strict=strict)
    if not records:
        if ieee_mode:
            raise FileNotFoundError(
                f"Aucun CSV trouvé dans {results_dir} (mode IEEE)."
            )
        print(f"Aucun CSV trouvé dans {results_dir} ; aucune figure générée.")
        return

    requested_algorithms = _parse_algorithms(algorithms)
    if requested_algorithms:
        algorithms_to_plot, duplicates = _unique_algorithms(requested_algorithms)
        if duplicates:
            duplicates_list = ", ".join(sorted(set(duplicates)))
            print(
                "Algorithmes dupliqués ignorés : "
                f"{duplicates_list}."
            )
    else:
        discovered = [
            str(record.get("algorithm") or "unknown")
            for record in records
        ]
        discovered_unique, _duplicates = _unique_algorithms(discovered)
        if "mixra_opt" in discovered_unique:
            algorithms_to_plot = ["mixra_opt"]
        else:
            algorithms_to_plot = discovered_unique

    if not algorithms_to_plot:
        print("Aucun algorithme trouvé ; aucune distribution générée.")
        return

    for algo in algorithms_to_plot:
        algo_records = [record for record in records if str(record.get("algorithm") or "unknown") == algo]
        if not algo_records:
            if ieee_mode:
                raise ValueError(
                    f"Aucune donnée pour l'algorithme {algo} (mode IEEE)."
                )
            print(f"Aucune donnée pour l'algorithme {algo}.")
            continue
        for metric, label in METRICS.items():
            _plot_distribution(algo_records, metric, label, algo, figures_dir)


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
        help="Répertoire de sortie des figures",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        help=(
            "Algorithme(s) à tracer (option répétable, liste séparée par des virgules). "
            "Par défaut, mixra_opt est utilisé si disponible afin d'éviter les figures isolées par algorithme."
        ),
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
    generate_distributions(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        algorithms=args.algorithm,
        strict=args.strict,
        ieee_mode=args.ieee,
    )


if __name__ == "__main__":
    main()
