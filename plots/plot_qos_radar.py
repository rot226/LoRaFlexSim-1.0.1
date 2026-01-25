"""Trace des radars QoS normalisés (DER, PDR, collisions, énergie, SNIR).

Le script produit deux radars pour un algorithme donné (SNIR activé/désactivé).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.plot_step1_results import (  # noqa: E402
    SNIR_COLORS,
    _apply_ieee_style,
    _detect_snir_state,
    _normalize_algorithm_name,
    _select_signal_mean,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "step1" / "radar"

METRICS = [
    ("DER", "DER (probability)"),
    ("PDR", "PDR (probability)"),
    ("collisions", "Collisions (probability)"),
    ("avg_energy_per_node_J", "Mean energy (J)"),
    ("snir_mean", "Mean SNIR/SNR (dB)"),
]

SNIR_STATES = ("snir_on", "snir_off")
SNIR_LABELS = {
    "snir_on": "SNIR enabled",
    "snir_off": "SNIR disabled",
}


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _extract_energy(row: Mapping[str, Any]) -> float | None:
    for key in (
        "avg_energy_per_node_J",
        "energy_per_node",
        "energy_nodes_J",
        "energy_J",
        "energy_j",
        "energy_mean_J",
    ):
        value = _parse_float(row.get(key))
        if value is not None:
            return value
    return None


def _load_qos_records(results_dir: Path, strict: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            if strict:
                required_columns = {"snir_state", "snir_mean"}
                fieldnames = set(reader.fieldnames or [])
                if not required_columns.issubset(fieldnames):
                    continue
            for row in reader:
                snir_state, snir_detected = _detect_snir_state(row)
                if not snir_detected or snir_state is None:
                    continue
                snir_candidate = row.get("snir_mean") or row.get("SNIR")
                snr_candidate = row.get("snr_mean") or row.get("SNR") or row.get("snr")
                algorithm = _normalize_algorithm_name(row.get("algorithm"))
                if not algorithm:
                    algorithm = (
                        _normalize_algorithm_name(csv_path.parent.name)
                        or csv_path.parent.name
                    )
                use_snir = True if snir_state == "snir_on" else False if snir_state == "snir_off" else None
                record: Dict[str, Any] = {
                    "algorithm": algorithm,
                    "snir_state": snir_state,
                    "DER": _parse_float(row.get("DER")),
                    "PDR": _parse_float(row.get("PDR")),
                    "collisions": _parse_float(row.get("collisions")),
                    "avg_energy_per_node_J": _extract_energy(row),
                    "snir_mean": _parse_float(snir_candidate),
                    "snr_mean": _parse_float(snr_candidate),
                    "use_snir": use_snir,
                }
                records.append(record)
    return records


def _mean(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return statistics.mean(cleaned)


def _aggregate_by_state(
    records: Sequence[Mapping[str, Any]],
    algorithm: str,
) -> Dict[str, Dict[str, float | None]]:
    metrics_by_state: Dict[str, Dict[str, float | None]] = {}
    for state in SNIR_STATES:
        state_records = [
            r
            for r in records
            if str(r.get("algorithm") or "").lower() == algorithm.lower()
            and r.get("snir_state") == state
        ]
        if not state_records:
            continue
        metrics: Dict[str, float | None] = {}
        for metric_key, _label in METRICS:
            if metric_key == "snir_mean":
                values = [_select_signal_mean(r)[0] for r in state_records]
            else:
                values = [r.get(metric_key) for r in state_records]
            metrics[metric_key] = _mean(values)
        metrics_by_state[state] = metrics
    return metrics_by_state


def _compute_normalization(
    metrics_by_state: Dict[str, Dict[str, float | None]]
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for metric_key, _ in METRICS:
        values: List[float] = []
        for metrics in metrics_by_state.values():
            value = metrics.get(metric_key)
            if value is not None:
                values.append(value)
        if values:
            bounds[metric_key] = (min(values), max(values))
        else:
            bounds[metric_key] = (0.0, 0.0)
    return bounds


def _normalize(value: float | None, bounds: Tuple[float, float]) -> float | None:
    if value is None:
        return None
    min_val, max_val = bounds
    if math.isclose(min_val, max_val):
        return 0.5
    return (value - min_val) / (max_val - min_val)


def _plot_radar(
    algorithm: str,
    state: str,
    metrics: Dict[str, float | None],
    bounds: Dict[str, Tuple[float, float]],
    figures_dir: Path,
) -> None:
    labels = [label for _, label in METRICS]
    metric_keys = [key for key, _ in METRICS]
    angles = [n / len(metric_keys) * 2 * math.pi for n in range(len(metric_keys))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.8, 6.2), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True, linestyle=":", linewidth=0.8)

    normalized = [_normalize(metrics.get(key), bounds[key]) for key in metric_keys]
    if any(value is None for value in normalized):
        return
    values = [float(value) for value in normalized]
    values += values[:1]
    color = SNIR_COLORS.get(state, "#333333")
    ax.plot(angles, values, color=color, linewidth=2, label=SNIR_LABELS.get(state, state))
    ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_title(f"QoS radar – {algorithm} ({SNIR_LABELS.get(state, state)})", y=1.1)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"qos_radar_{algorithm.replace(' ', '_')}_{state}.png"
    plt.subplots_adjust(top=0.80)
    fig.savefig(figures_dir / output_name, dpi=300)
    plt.close(fig)


def generate_qos_radars(
    results_dir: Path,
    figures_dir: Path,
    strict: bool,
    algorithm: str,
    ieee_mode: bool,
) -> None:
    records = _load_qos_records(results_dir, strict=strict)
    if not records:
        if ieee_mode:
            raise FileNotFoundError(
                f"Aucun CSV trouvé dans {results_dir} (mode IEEE)."
            )
        print(f"Aucun CSV trouvé dans {results_dir} ; aucun radar généré.")
        return

    _apply_ieee_style()
    metrics_by_state = _aggregate_by_state(records, algorithm=algorithm)
    if not metrics_by_state:
        if ieee_mode:
            raise ValueError(
                f"Aucun enregistrement trouvé pour l'algorithme {algorithm} (mode IEEE)."
            )
        print(f"Aucun enregistrement trouvé pour l'algorithme {algorithm}.")
        return
    bounds = _compute_normalization(metrics_by_state)

    for state, metrics in metrics_by_state.items():
        _plot_radar(algorithm, state, metrics, bounds, figures_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant les CSV QoS (étape 1)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire de sortie des radars",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Ignore les CSV sans colonnes SNIR explicites (snir_state, snir_mean).",
    )
    parser.add_argument(
        "--algorithm",
        default="mixra_opt",
        help="Algorithme à tracer (par défaut : mixra_opt).",
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
    generate_qos_radars(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        strict=args.strict,
        algorithm=args.algorithm,
        ieee_mode=args.ieee,
    )


if __name__ == "__main__":
    main()
