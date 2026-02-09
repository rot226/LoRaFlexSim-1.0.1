"""Génère les figures RLS1–RLS8 (format IEEE) à partir des CSV IWCMC RL static."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # pragma: no cover - matplotlib optionnel
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from article_c.common.plotting_style import (
    FIGURE_MARGINS,
    LEGEND_STYLE,
    TIGHT_LAYOUT_RECT,
    apply_ieee_style,
    apply_output_fonttype,
    legend_bbox_to_anchor,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "iwcmc" / "rl_static"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "IWCMC" / "rl_static" / "plots" / "figures"


@dataclass(frozen=True)
class Record:
    algorithm: str
    num_nodes: int
    snir_state: str
    replication_index: int
    der: float
    throughput_bps: float
    snir_mean: float
    collision_rate: float
    jain_index: float
    avg_energy_per_node_J: float
    sf_distribution: Dict[int, float]


ALGO_LABELS = {
    "adr pur": "ADR",
    "adr": "ADR",
    "mixra-h": "MixRA-H",
    "mixra_h": "MixRA-H",
    "mixra-opt": "MixRA-Opt",
    "mixra_opt": "MixRA-Opt",
    "ucb1": "UCB1",
}

COLORS = {
    "UCB1": "#1f77b4",
    "ADR": "#ff7f0e",
    "MixRA-H": "#2ca02c",
    "MixRA-Opt": "#d62728",
}
MARKERS = {
    "UCB1": "o",
    "ADR": "s",
    "MixRA-H": "^",
    "MixRA-Opt": "D",
}


def _normalize_algo(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Inconnu"
    key = text.lower().replace(" ", "_")
    return ALGO_LABELS.get(key, text)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_sf_distribution(row: Mapping[str, Any]) -> Dict[int, float]:
    dist: Dict[int, float] = {}
    for sf in range(7, 13):
        key = f"sf_distribution__{sf}"
        if key in row:
            dist[sf] = _parse_float(row.get(key), 0.0)
    return dist


def _load_records(results_dir: Path, snir_state: str | None) -> List[Record]:
    records: List[Record] = []
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                state = str(row.get("snir_state") or "").strip().lower()
                if snir_state and state != snir_state:
                    continue
                record = Record(
                    algorithm=_normalize_algo(row.get("algorithm") or csv_path.parent.name),
                    num_nodes=_parse_int(row.get("num_nodes")),
                    snir_state=state or "snir_unknown",
                    replication_index=_parse_int(row.get("replication_index"), 1),
                    der=_parse_float(row.get("DER")),
                    throughput_bps=_parse_float(row.get("throughput_bps")),
                    snir_mean=_parse_float(row.get("snir_mean")),
                    collision_rate=_parse_float(row.get("collision_rate")),
                    jain_index=_parse_float(row.get("jain_index")),
                    avg_energy_per_node_J=_parse_float(row.get("avg_energy_per_node_J")),
                    sf_distribution=_extract_sf_distribution(row),
                )
                records.append(record)
    return records


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    mean = fmean(values)
    std = pstdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, margin


def _group_by_algorithm(records: Iterable[Record]) -> Dict[str, List[Record]]:
    grouped: Dict[str, List[Record]] = {}
    for record in records:
        grouped.setdefault(record.algorithm, []).append(record)
    return grouped


def _style_for(algo: str) -> dict[str, Any]:
    return {
        "color": COLORS.get(algo, "#7f7f7f"),
        "marker": MARKERS.get(algo, "o"),
        "label": algo,
    }


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), dpi=300)


def _plot_metric_vs_nodes(
    records: Sequence[Record],
    metric: str,
    ylabel: str,
    output_path: Path,
    *,
    title: str,
) -> None:
    grouped = _group_by_algorithm(records)
    apply_ieee_style()
    fig, ax = plt.subplots()

    for algo, algo_records in grouped.items():
        nodes = sorted({rec.num_nodes for rec in algo_records})
        means: List[float] = []
        margins: List[float] = []
        for num_nodes in nodes:
            values = [getattr(rec, metric) for rec in algo_records if rec.num_nodes == num_nodes]
            mean, margin = _mean_ci(values)
            means.append(mean)
            margins.append(margin)
        style = _style_for(algo)
        ax.errorbar(nodes, means, yerr=margins, capsize=3, **style)

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_snir_vs_iterations(
    records: Sequence[Record],
    output_path: Path,
    *,
    title: str,
) -> None:
    grouped = _group_by_algorithm(records)
    apply_ieee_style()
    fig, ax = plt.subplots()

    for algo, algo_records in grouped.items():
        ordered = sorted(algo_records, key=lambda rec: rec.replication_index)
        x_vals = [rec.replication_index for rec in ordered]
        y_vals = [rec.snir_mean for rec in ordered]
        ax.plot(x_vals, y_vals, **_style_for(algo))

    ax.set_xlabel("Itération")
    ax.set_ylabel("SNIR moyen (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_convergence(
    records: Sequence[Record],
    output_path: Path,
    *,
    title: str,
) -> None:
    grouped = _group_by_algorithm(records)
    apply_ieee_style()
    fig, ax = plt.subplots()

    for algo, algo_records in grouped.items():
        ordered = sorted(algo_records, key=lambda rec: rec.replication_index)
        cumulative: List[float] = []
        acc = 0.0
        for idx, rec in enumerate(ordered, start=1):
            acc += rec.der
            cumulative.append(acc / idx)
        x_vals = [rec.replication_index for rec in ordered]
        ax.plot(x_vals, cumulative, **_style_for(algo))

    ax.set_xlabel("Itération")
    ax.set_ylabel("DER cumulée")
    ax.set_title(title)
    ax.grid(True, alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_sf_distribution(
    records: Sequence[Record],
    output_path: Path,
    *,
    title: str,
) -> None:
    grouped = _group_by_algorithm(records)
    apply_ieee_style()
    fig, ax = plt.subplots()

    sfs = list(range(7, 13))
    x_positions = list(range(len(sfs)))
    width = 0.2
    offset = 0.0
    for algo, algo_records in grouped.items():
        totals: Dict[int, float] = {sf: 0.0 for sf in sfs}
        for rec in algo_records:
            for sf in sfs:
                totals[sf] += rec.sf_distribution.get(sf, 0.0)
        count = max(1, len(algo_records))
        averages = [totals[sf] / count for sf in sfs]
        positions = [x + offset for x in x_positions]
        ax.bar(positions, averages, width=width, color=COLORS.get(algo, "#7f7f7f"), label=algo)
        offset += width

    ax.set_xlabel("Facteur d'étalement (SF)")
    ax.set_ylabel("Nombre moyen de nœuds")
    ax.set_title(title)
    ax.set_xticks([x + width for x in x_positions])
    ax.set_xticklabels([str(sf) for sf in sfs])
    ax.grid(True, axis="y", alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_path)
    plt.close(fig)


def _select_records_for_nodes(records: Sequence[Record], num_nodes: int | None) -> List[Record]:
    if num_nodes is None:
        return list(records)
    return [record for record in records if record.num_nodes == num_nodes]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire des CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des figures.",
    )
    parser.add_argument(
        "--snir-state",
        choices=["snir_on", "snir_off", "snir_unknown"],
        default="snir_on",
        help="Filtre sur l'état SNIR.",
    )
    parser.add_argument(
        "--snir-iterations-nodes",
        type=int,
        default=None,
        help="Taille de réseau utilisée pour SNIR vs itérations/convergence.",
    )
    parser.add_argument(
        "--sf-distribution-nodes",
        type=int,
        default=None,
        help="Taille de réseau utilisée pour la distribution SF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if plt is None:
        raise RuntimeError("matplotlib est requis pour générer les figures.")
    parser = _build_parser()
    args = parser.parse_args(argv)

    apply_output_fonttype()
    records = _load_records(args.results_dir, args.snir_state)
    if not records:
        raise RuntimeError("Aucun CSV trouvé pour les figures RLS.")

    snir_records = _select_records_for_nodes(records, args.snir_iterations_nodes)
    sf_records = _select_records_for_nodes(records, args.sf_distribution_nodes)

    _plot_metric_vs_nodes(
        records,
        "der",
        "DER",
        args.output_dir / "RLS1_der_vs_nodes.pdf",
        title="RLS1 – DER vs nœuds",
    )
    _plot_metric_vs_nodes(
        records,
        "throughput_bps",
        "Débit (bps)",
        args.output_dir / "RLS2_throughput_vs_nodes.pdf",
        title="RLS2 – Débit vs nœuds",
    )
    _plot_snir_vs_iterations(
        snir_records,
        args.output_dir / "RLS3_snir_vs_iterations.pdf",
        title="RLS3 – SNIR vs itérations",
    )
    _plot_metric_vs_nodes(
        records,
        "collision_rate",
        "Taux de collisions",
        args.output_dir / "RLS4_collision_rate_vs_nodes.pdf",
        title="RLS4 – Collisions vs nœuds",
    )
    _plot_sf_distribution(
        sf_records,
        args.output_dir / "RLS5_sf_distribution.pdf",
        title="RLS5 – Distribution des SF",
    )
    _plot_convergence(
        snir_records,
        args.output_dir / "RLS6_convergence.pdf",
        title="RLS6 – Convergence DER",
    )
    _plot_metric_vs_nodes(
        records,
        "jain_index",
        "Indice de Jain",
        args.output_dir / "RLS7_equite_vs_nodes.pdf",
        title="RLS7 – Équité vs nœuds",
    )
    _plot_metric_vs_nodes(
        records,
        "avg_energy_per_node_J",
        "Énergie moyenne par nœud (J)",
        args.output_dir / "RLS8_energy_vs_nodes.pdf",
        title="RLS8 – Énergie vs nœuds",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
