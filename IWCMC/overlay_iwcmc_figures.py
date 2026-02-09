"""Superpose les courbes IWCMC (S1–S8, RLS1–RLS8, RLM1–RLM8)."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plotting_style import (
    FIGURE_MARGINS,
    LEGEND_STYLE,
    TIGHT_LAYOUT_RECT,
    apply_ieee_style,
    apply_output_fonttype,
    legend_bbox_to_anchor,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SNIR_DATA_DIR = ROOT_DIR / "IWCMC" / "snir_static" / "data"
DEFAULT_RLS_RESULTS_DIR = ROOT_DIR / "results" / "iwcmc" / "rl_static"
DEFAULT_RLM_RESULTS_DIR = ROOT_DIR / "results" / "iwcmc" / "rl_mobile"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "IWCMC" / "figures" / "overlays"

SCENARIOS_SNIR = [f"S{idx}" for idx in range(1, 9)]

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


@dataclass(frozen=True)
class RLRecord:
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
    mobility_model: str | None = None
    mobility_speed_min: float | None = None
    mobility_speed_max: float | None = None


@dataclass(frozen=True)
class CampaignStyle:
    name: str
    linestyle: str


CAMPAIGNS = (
    CampaignStyle("Statique", "-"),
    CampaignStyle("Mobile", "--"),
)


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


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    mean = fmean(values)
    std = pstdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, margin


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _style_for(algo: str, campaign: CampaignStyle) -> dict[str, Any]:
    return {
        "color": COLORS.get(algo, "#7f7f7f"),
        "marker": MARKERS.get(algo, "o"),
        "linestyle": campaign.linestyle,
    }


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), dpi=300)


def _load_snir_static_means(data_dir: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS_SNIR:
        csv_path = data_dir / f"{scenario}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        required = {"cluster_id", "algorithm", "pdr_achieved"}
        if not required.issubset(set(df.columns)):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}.")
        if "figure" in df.columns:
            df = df[df["figure"] == scenario]
        if df.empty:
            continue
        grouped = df.groupby("algorithm", as_index=False)["pdr_achieved"].mean()
        results[scenario] = {
            _normalize_algo(row["algorithm"]): float(row["pdr_achieved"])
            for _, row in grouped.iterrows()
        }
    return results


def _plot_snir_overlay(data_dir: Path, output_dir: Path) -> None:
    snir_means = _load_snir_static_means(data_dir)
    if not snir_means:
        raise RuntimeError("Aucune donnée SNIR statique trouvée pour S1–S8.")
    algorithms = sorted({algo for values in snir_means.values() for algo in values})
    apply_ieee_style()
    fig, ax = plt.subplots()

    x_positions = list(range(1, len(SCENARIOS_SNIR) + 1))
    for algo in algorithms:
        y_values = [snir_means.get(scenario, {}).get(algo, float("nan")) for scenario in SCENARIOS_SNIR]
        ax.plot(
            x_positions,
            y_values,
            label=algo,
            color=COLORS.get(algo, "#7f7f7f"),
            marker=MARKERS.get(algo, "o"),
        )

    ax.set_xticks(x_positions, SCENARIOS_SNIR)
    ax.set_xlabel("Scénario SNIR")
    ax.set_ylabel("PDR moyenne")
    ax.set_title("S1–S8 – PDR moyenne par algorithme")
    ax.grid(True, alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_dir / "S1_S8_overlay_pdr.pdf")
    plt.close(fig)


def _load_rl_records(
    results_dir: Path,
    snir_state: str | None,
    *,
    mobility_model: str | None = None,
    speed_range: tuple[float, float] | None = None,
) -> List[RLRecord]:
    records: List[RLRecord] = []
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                state = str(row.get("snir_state") or "").strip().lower()
                if snir_state and state != snir_state:
                    continue
                model = str(row.get("mobility_model") or "").strip().lower() or None
                if mobility_model and model != mobility_model:
                    continue
                min_speed = _parse_float(row.get("mobility_speed_min"))
                max_speed = _parse_float(row.get("mobility_speed_max"))
                if speed_range and (min_speed, max_speed) != speed_range:
                    continue
                record = RLRecord(
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
                    mobility_model=model,
                    mobility_speed_min=min_speed,
                    mobility_speed_max=max_speed,
                )
                records.append(record)
    return records


def _group_by_algorithm(records: Iterable[RLRecord]) -> Dict[str, List[RLRecord]]:
    grouped: Dict[str, List[RLRecord]] = {}
    for record in records:
        grouped.setdefault(record.algorithm, []).append(record)
    return grouped


def _select_records_for_nodes(records: Sequence[RLRecord], num_nodes: int | None) -> List[RLRecord]:
    if num_nodes is None:
        return list(records)
    return [record for record in records if record.num_nodes == num_nodes]


def _plot_metric_overlay(
    static_records: Sequence[RLRecord],
    mobile_records: Sequence[RLRecord],
    metric: str,
    ylabel: str,
    output_path: Path,
    *,
    title: str,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()

    for campaign, records in zip(CAMPAIGNS, (static_records, mobile_records)):
        grouped = _group_by_algorithm(records)
        for algo, algo_records in grouped.items():
            nodes = sorted({rec.num_nodes for rec in algo_records})
            means: List[float] = []
            margins: List[float] = []
            for num_nodes in nodes:
                values = [getattr(rec, metric) for rec in algo_records if rec.num_nodes == num_nodes]
                mean, margin = _mean_ci(values)
                means.append(mean)
                margins.append(margin)
            style = _style_for(algo, campaign)
            ax.errorbar(
                nodes,
                means,
                yerr=margins,
                capsize=3,
                label=f"{algo} ({campaign.name})",
                **style,
            )

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


def _plot_snir_iterations_overlay(
    static_records: Sequence[RLRecord],
    mobile_records: Sequence[RLRecord],
    output_path: Path,
    *,
    title: str,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()

    for campaign, records in zip(CAMPAIGNS, (static_records, mobile_records)):
        grouped = _group_by_algorithm(records)
        for algo, algo_records in grouped.items():
            ordered = sorted(algo_records, key=lambda rec: rec.replication_index)
            x_vals = [rec.replication_index for rec in ordered]
            y_vals = [rec.snir_mean for rec in ordered]
            style = _style_for(algo, campaign)
            ax.plot(x_vals, y_vals, label=f"{algo} ({campaign.name})", **style)

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


def _plot_convergence_overlay(
    static_records: Sequence[RLRecord],
    mobile_records: Sequence[RLRecord],
    output_path: Path,
    *,
    title: str,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()

    for campaign, records in zip(CAMPAIGNS, (static_records, mobile_records)):
        grouped = _group_by_algorithm(records)
        for algo, algo_records in grouped.items():
            ordered = sorted(algo_records, key=lambda rec: rec.replication_index)
            cumulative: List[float] = []
            acc = 0.0
            for idx, rec in enumerate(ordered, start=1):
                acc += rec.der
                cumulative.append(acc / idx)
            x_vals = [rec.replication_index for rec in ordered]
            style = _style_for(algo, campaign)
            ax.plot(x_vals, cumulative, label=f"{algo} ({campaign.name})", **style)

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


def _plot_sf_distribution_overlay(
    static_records: Sequence[RLRecord],
    mobile_records: Sequence[RLRecord],
    output_path: Path,
    *,
    title: str,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()

    sfs = list(range(7, 13))
    for campaign, records in zip(CAMPAIGNS, (static_records, mobile_records)):
        grouped = _group_by_algorithm(records)
        for algo, algo_records in grouped.items():
            totals: Dict[int, float] = {sf: 0.0 for sf in sfs}
            for rec in algo_records:
                for sf in sfs:
                    totals[sf] += rec.sf_distribution.get(sf, 0.0)
            count = max(1, len(algo_records))
            averages = [totals[sf] / count for sf in sfs]
            style = _style_for(algo, campaign)
            ax.plot(
                sfs,
                averages,
                label=f"{algo} ({campaign.name})",
                **style,
            )

    ax.set_xlabel("Facteur d'étalement (SF)")
    ax.set_ylabel("Nombre moyen de nœuds")
    ax.set_title(title)
    ax.grid(True, alpha=0.6)
    legend = ax.legend(**LEGEND_STYLE)
    if legend:
        legend.set_bbox_to_anchor(legend_bbox_to_anchor(legend=legend))
    fig.subplots_adjust(**FIGURE_MARGINS)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_rl_overlays(
    static_records: Sequence[RLRecord],
    mobile_records: Sequence[RLRecord],
    output_dir: Path,
    *,
    snir_iterations_nodes: int | None,
    sf_distribution_nodes: int | None,
) -> None:
    if not static_records and not mobile_records:
        raise RuntimeError("Aucun CSV trouvé pour les overlays RL.")

    snir_static = _select_records_for_nodes(static_records, snir_iterations_nodes)
    snir_mobile = _select_records_for_nodes(mobile_records, snir_iterations_nodes)
    sf_static = _select_records_for_nodes(static_records, sf_distribution_nodes)
    sf_mobile = _select_records_for_nodes(mobile_records, sf_distribution_nodes)

    _plot_metric_overlay(
        static_records,
        mobile_records,
        "der",
        "DER",
        output_dir / "RLS1_RLM1_der_vs_nodes.pdf",
        title="RLS1/RLM1 – DER vs nœuds",
    )
    _plot_metric_overlay(
        static_records,
        mobile_records,
        "throughput_bps",
        "Débit (bps)",
        output_dir / "RLS2_RLM2_throughput_vs_nodes.pdf",
        title="RLS2/RLM2 – Débit vs nœuds",
    )
    _plot_snir_iterations_overlay(
        snir_static,
        snir_mobile,
        output_dir / "RLS3_RLM3_snir_vs_iterations.pdf",
        title="RLS3/RLM3 – SNIR vs itérations",
    )
    _plot_metric_overlay(
        static_records,
        mobile_records,
        "collision_rate",
        "Taux de collisions",
        output_dir / "RLS4_RLM4_collision_rate_vs_nodes.pdf",
        title="RLS4/RLM4 – Collisions vs nœuds",
    )
    _plot_sf_distribution_overlay(
        sf_static,
        sf_mobile,
        output_dir / "RLS5_RLM5_sf_distribution.pdf",
        title="RLS5/RLM5 – Distribution des SF",
    )
    _plot_convergence_overlay(
        snir_static,
        snir_mobile,
        output_dir / "RLS6_RLM6_convergence.pdf",
        title="RLS6/RLM6 – Convergence DER",
    )
    _plot_metric_overlay(
        static_records,
        mobile_records,
        "jain_index",
        "Indice de Jain",
        output_dir / "RLS7_RLM7_equite_vs_nodes.pdf",
        title="RLS7/RLM7 – Équité vs nœuds",
    )
    _plot_metric_overlay(
        static_records,
        mobile_records,
        "avg_energy_per_node_J",
        "Énergie moyenne par nœud (J)",
        output_dir / "RLS8_RLM8_energy_vs_nodes.pdf",
        title="RLS8/RLM8 – Énergie vs nœuds",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snir-data-dir",
        type=Path,
        default=DEFAULT_SNIR_DATA_DIR,
        help="Répertoire des CSV SNIR statique (S1–S8).",
    )
    parser.add_argument(
        "--rls-results-dir",
        type=Path,
        default=DEFAULT_RLS_RESULTS_DIR,
        help="Répertoire des CSV RL statique.",
    )
    parser.add_argument(
        "--rlm-results-dir",
        type=Path,
        default=DEFAULT_RLM_RESULTS_DIR,
        help="Répertoire des CSV RL mobile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des overlays.",
    )
    parser.add_argument(
        "--snir-state",
        choices=["snir_on", "snir_off", "snir_unknown"],
        default="snir_on",
        help="Filtre sur l'état SNIR pour les courbes RL.",
    )
    parser.add_argument(
        "--mobility-model",
        default=None,
        help="Filtre sur le modèle de mobilité pour RL mobile.",
    )
    parser.add_argument(
        "--mobility-speed-min",
        type=float,
        default=None,
        help="Vitesse min pour RL mobile (filtre).",
    )
    parser.add_argument(
        "--mobility-speed-max",
        type=float,
        default=None,
        help="Vitesse max pour RL mobile (filtre).",
    )
    parser.add_argument(
        "--snir-iterations-nodes",
        type=int,
        default=None,
        help="Taille de réseau pour les courbes SNIR/convergence.",
    )
    parser.add_argument(
        "--sf-distribution-nodes",
        type=int,
        default=None,
        help="Taille de réseau pour la distribution SF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    apply_output_fonttype()
    _ensure_output_dir(args.output_dir)

    _plot_snir_overlay(args.snir_data_dir, args.output_dir)

    speed_range = None
    if args.mobility_speed_min is not None and args.mobility_speed_max is not None:
        speed_range = (args.mobility_speed_min, args.mobility_speed_max)

    static_records = _load_rl_records(args.rls_results_dir, args.snir_state)
    mobile_records = _load_rl_records(
        args.rlm_results_dir,
        args.snir_state,
        mobility_model=(args.mobility_model.lower() if args.mobility_model else None),
        speed_range=speed_range,
    )
    _plot_rl_overlays(
        static_records,
        mobile_records,
        args.output_dir,
        snir_iterations_nodes=args.snir_iterations_nodes,
        sf_distribution_nodes=args.sf_distribution_nodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
