"""Génère les visualisations du banc QoS par clusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:  # pragma: no cover - dépend de l'environnement de test
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - permet une dégradation élégante sans numpy réel
    plt = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "qos_clusters"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "qos_clusters"

__all__ = [
    "generate_plots",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_FIGURES_DIR",
]


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _load_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return records
    for algo_dir in sorted(results_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        for csv_path in sorted(algo_dir.glob("*.csv")):
            with csv_path.open("r", encoding="utf8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    record: Dict[str, Any] = {
                        "csv_path": csv_path,
                        "algorithm": row.get("algorithm", algo_dir.name),
                        "num_nodes": int(float(row.get("num_nodes", "0") or 0)),
                        "packet_interval_s": float(row.get("packet_interval_s", "0") or 0),
                        "PDR": _parse_float(row.get("PDR")),
                        "DER": _parse_float(row.get("DER")),
                        "throughput_bps": _parse_float(row.get("throughput_bps")),
                        "avg_energy_per_node_J": _parse_float(row.get("avg_energy_per_node_J")),
                        "jain_index": _parse_float(row.get("jain_index")),
                        "mixra_solver": row.get("mixra_solver"),
                    }
                    cluster_pdr: Dict[int, float] = {}
                    cluster_targets: Dict[int, float] = {}
                    for key, value in row.items():
                        if key.startswith("qos_cluster_pdr__"):
                            cluster_id = int(key.split("__")[-1])
                            cluster_pdr[cluster_id] = _parse_float(value)
                        elif key.startswith("qos_cluster_targets__"):
                            cluster_id = int(key.split("__")[-1])
                            cluster_targets[cluster_id] = _parse_float(value)
                        elif key.startswith("sf_distribution__"):
                            sf = int(key.split("__")[-1])
                            record.setdefault("sf_distribution", {})[sf] = _parse_float(value)
                    record["cluster_pdr"] = cluster_pdr
                    record["cluster_targets"] = cluster_targets
                    if "throughput_sf_channel_json" in row:
                        try:
                            data = json.loads(row["throughput_sf_channel_json"])
                            throughput_map = {
                                int(sf): {int(ch): float(val) for ch, val in channel_map.items()}
                                for sf, channel_map in data.items()
                            }
                            record["throughput_sf_channel"] = throughput_map
                        except Exception:
                            record["throughput_sf_channel"] = {}
                    if "snr_cdf_json" in row:
                        try:
                            record["snr_cdf"] = [
                                (float(point[0]), float(point[1])) for point in json.loads(row["snr_cdf_json"])
                            ]
                        except Exception:
                            record["snr_cdf"] = []
                    if "snr_histogram_json" in row:
                        try:
                            hist = json.loads(row["snr_histogram_json"])
                            record["snr_histogram"] = {float(bin_key): float(count) for bin_key, count in hist.items()}
                        except Exception:
                            record["snr_histogram"] = {}
                    records.append(record)
    return records


def _group_by(records: Iterable[Mapping[str, Any]], key: str) -> Dict[Any, List[Mapping[str, Any]]]:
    groups: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return groups


def _plot_metric_vs_nodes(
    records: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    filename_prefix: str,
    figures_dir: Path,
) -> None:
    if not records:
        return
    periods = sorted({record["packet_interval_s"] for record in records})
    algorithms = sorted({record["algorithm"] for record in records})
    for period in periods:
        fig, ax = plt.subplots(figsize=(6, 4))
        for algorithm in algorithms:
            data = [
                record
                for record in records
                if record["algorithm"] == algorithm and record["packet_interval_s"] == period
            ]
            data.sort(key=lambda item: item["num_nodes"])
            if not data:
                continue
            xs = [item["num_nodes"] for item in data]
            ys = [item.get(metric, 0.0) for item in data]
            ax.plot(xs, ys, marker="o", label=algorithm)
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel(ylabel)
        title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
        ax.set_title(f"{ylabel} – période {title_period} s")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = figures_dir / f"{filename_prefix}_tx_{title_period}.png"
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def _plot_pdr_clusters(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records:
        return
    clusters = sorted({cid for record in records for cid in record.get("cluster_pdr", {})})
    if not clusters:
        return
    periods = sorted({record["packet_interval_s"] for record in records})
    algorithms = sorted({record["algorithm"] for record in records})
    for period in periods:
        filtered = [r for r in records if r["packet_interval_s"] == period]
        if not filtered:
            continue
        fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
        if len(clusters) == 1:
            axes = [axes]
        for idx, cluster_id in enumerate(clusters):
            ax = axes[idx]
            for algorithm in algorithms:
                data = [r for r in filtered if r["algorithm"] == algorithm]
                data.sort(key=lambda item: item["num_nodes"])
                xs = []
                ys = []
                for item in data:
                    pdr_value = item.get("cluster_pdr", {}).get(cluster_id)
                    if pdr_value is None:
                        continue
                    xs.append(item["num_nodes"])
                    ys.append(pdr_value)
                if xs:
                    ax.plot(xs, ys, marker="o", label=algorithm)
            target = None
            for item in filtered:
                target = item.get("cluster_targets", {}).get(cluster_id)
                if target is not None:
                    break
            if target is not None:
                ax.axhline(target, color="black", linestyle="--", linewidth=1, label="Cible" if idx == 0 else None)
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Nœuds")
            if idx == 0:
                ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
        title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
        fig.suptitle(f"PDR par cluster – période {title_period} s")
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = figures_dir / f"pdr_clusters_tx_{title_period}.png"
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output, dpi=150)
        plt.close(fig)


def _select_worst_cases(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not records:
        return {}
    max_nodes = max(record["num_nodes"] for record in records)
    min_period = min(record["packet_interval_s"] for record in records)
    selected: Dict[str, Dict[str, Any]] = {}
    for algorithm in sorted({record["algorithm"] for record in records}):
        candidates = [
            record
            for record in records
            if record["algorithm"] == algorithm
            and record["num_nodes"] == max_nodes
            and record["packet_interval_s"] == min_period
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.get("csv_path", Path("")))
        selected[algorithm] = candidates[0]
    return selected


def _plot_snr_cdf(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    selected = _select_worst_cases(records)
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for algorithm, record in selected.items():
        cdf = record.get("snr_cdf", [])
        if not cdf:
            continue
        xs = [point[0] for point in cdf]
        ys = [point[1] for point in cdf]
        ax.plot(xs, ys, label=algorithm)
    ax.set_xlabel("SNIR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF du SNIR – charge maximale")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "snr_cdf_max_load.png", dpi=150)
    plt.close(fig)


def _plot_sf_histogram(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    selected = _select_worst_cases(records)
    if not selected:
        return
    algorithms = list(selected.keys())
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(6, 3 * len(algorithms)), sharex=True)
    if len(algorithms) == 1:
        axes = [axes]
    for ax, algorithm in zip(axes, algorithms):
        record = selected[algorithm]
        distribution = record.get("sf_distribution", {})
        if not distribution:
            continue
        sfs = sorted(distribution)
        values = [distribution[sf] for sf in sfs]
        ax.bar([str(sf) for sf in sfs], values, color="#4e79a7")
        ax.set_ylabel("Nœuds")
        ax.set_title(algorithm)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    axes[-1].set_xlabel("Spreading Factor")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "sf_histogram_max_load.png", dpi=150)
    plt.close(fig)


def _plot_heatmap(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    selected = _select_worst_cases(records)
    if not selected:
        return
    for algorithm, record in selected.items():
        throughput_map = record.get("throughput_sf_channel", {})
        if not throughput_map:
            continue
        sfs = sorted(throughput_map)
        channels = sorted({channel for mapping in throughput_map.values() for channel in mapping})
        if not sfs or not channels:
            continue
        matrix = [
            [throughput_map.get(sf, {}).get(channel, 0.0) for channel in channels]
            for sf in sfs
        ]
        fig, ax = plt.subplots(figsize=(1.2 * len(channels) + 3, 0.8 * len(sfs) + 2))
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels([str(ch) for ch in channels])
        ax.set_yticks(range(len(sfs)))
        ax.set_yticklabels([str(sf) for sf in sfs])
        ax.set_xlabel("Canal")
        ax.set_ylabel("SF")
        ax.set_title(f"Heatmap SF×canal – {algorithm}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Débit (bps)")
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(figures_dir / f"heatmap_sf_channel_{algorithm.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)


def generate_plots(results_dir: Path, figures_dir: Path, *, quiet: bool = False) -> bool:
    """Génère l'ensemble des figures à partir des CSV du banc QoS.

    Retourne ``True`` si au moins une figure a été produite.
    """

    if plt is None:
        if not quiet:
            print(
                "Matplotlib ou NumPy manquant : impossible de générer les graphiques. "
                "Installez les dépendances complètes pour activer cette étape."
            )
        return False

    records = _load_records(results_dir)
    if not records:
        if not quiet:
            print("Aucun résultat trouvé, aucune figure générée.")
        return False
    _plot_pdr_clusters(records, figures_dir)
    _plot_metric_vs_nodes(records, "DER", "DER", "der", figures_dir)
    _plot_metric_vs_nodes(records, "throughput_bps", "Débit (bps)", "throughput", figures_dir)
    _plot_metric_vs_nodes(records, "avg_energy_per_node_J", "Énergie moyenne (J)", "energy", figures_dir)
    _plot_snr_cdf(records, figures_dir)
    _plot_sf_histogram(records, figures_dir)
    _plot_heatmap(records, figures_dir)
    if not quiet:
        print(f"Figures enregistrées dans {figures_dir}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Dossier contenant les CSV du banc QoS",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Dossier de sortie des figures",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="N'affiche pas les messages de progression",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_plots(args.results_dir, args.figures_dir, quiet=args.quiet)


if __name__ == "__main__":
    main()
