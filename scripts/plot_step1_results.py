"""Génère les figures de l'étape 1 à partir des CSV produits par Tâche 4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:  # pragma: no cover - dépend de l'environnement de test
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - permet de continuer même sans matplotlib
    plt = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures"

__all__ = ["generate_step1_figures", "DEFAULT_RESULTS_DIR", "DEFAULT_FIGURES_DIR"]

STATE_LABELS = {True: "snir_on", False: "snir_off", None: "snir_unknown"}


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _detect_snir(row: Mapping[str, Any], path: Path) -> bool | None:
    candidates = [
        row.get("use_snir"),
        row.get("channel__use_snir"),
        row.get("snir_enabled"),
        row.get("snir"),
    ]
    for candidate in candidates:
        parsed = _parse_bool(candidate)
        if parsed is not None:
            return parsed
    for part in path.parts:
        if "snir" in part.lower():
            return True
    return None


def _load_step1_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cluster_pdr: Dict[int, float] = {}
                cluster_targets: Dict[int, float] = {}
                for key, value in row.items():
                    if key.startswith("qos_cluster_pdr__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_pdr[cluster_id] = _parse_float(value)
                    elif key.startswith("qos_cluster_targets__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_targets[cluster_id] = _parse_float(value)

                record: Dict[str, Any] = {
                    "csv_path": csv_path,
                    "algorithm": row.get("algorithm", csv_path.parent.name),
                    "num_nodes": int(float(row.get("num_nodes", "0") or 0)),
                    "packet_interval_s": float(row.get("packet_interval_s", "0") or 0),
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "collisions": int(float(row.get("collisions", "0") or 0)),
                    "collisions_snir": int(float(row.get("collisions_snir", "0") or 0)),
                    "jain_index": _parse_float(row.get("jain_index")),
                    "throughput_bps": _parse_float(row.get("throughput_bps")),
                    "cluster_pdr": cluster_pdr,
                    "cluster_targets": cluster_targets,
                }
                snir_flag = _detect_snir(row, csv_path)
                record["use_snir"] = snir_flag
                record["snir_state"] = STATE_LABELS.get(snir_flag, "snir_unknown")
                records.append(record)
    return records


def _plot_global_metric(
    records: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    filename_prefix: str,
    figures_dir: Path,
) -> None:
    if not records or plt is None:
        return
    periods = sorted({r["packet_interval_s"] for r in records})
    algorithms = sorted({r["algorithm"] for r in records})
    snir_states = sorted({r.get("snir_state") for r in records}, key=str)
    for state in snir_states:
        state_records = [r for r in records if r.get("snir_state") == state]
        if not state_records:
            continue
        label_state = "SNIR activé" if state == "snir_on" else "SNIR désactivé" if state == "snir_off" else "SNIR inconnu"
        suffix_state = state or "snir_unknown"
        for period in periods:
            fig, ax = plt.subplots(figsize=(6, 4))
            for algorithm in algorithms:
                data = [
                    r
                    for r in state_records
                    if r["algorithm"] == algorithm and r["packet_interval_s"] == period
                ]
                data.sort(key=lambda item: item["num_nodes"])
                xs = [item["num_nodes"] for item in data]
                ys = [item.get(metric, 0.0) for item in data]
                if xs:
                    ax.plot(xs, ys, marker="o", label=algorithm)
            ax.set_xlabel("Nombre de nœuds")
            ax.set_ylabel(ylabel)
            title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            ax.set_title(f"{ylabel} – {label_state} – période {title_period} s")
            ax.grid(True, linestyle=":", alpha=0.5)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"step1_{filename_prefix}_{suffix_state}_tx_{title_period}.png"
            fig.tight_layout()
            fig.savefig(output, dpi=150)
            plt.close(fig)


def _plot_cluster_pdr(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return
    clusters = sorted({cid for r in records for cid in r.get("cluster_pdr", {})})
    if not clusters:
        return
    periods = sorted({r["packet_interval_s"] for r in records})
    algorithms = sorted({r["algorithm"] for r in records})
    snir_states = sorted({r.get("snir_state") for r in records}, key=str)
    for state in snir_states:
        state_records = [r for r in records if r.get("snir_state") == state]
        if not state_records:
            continue
        label_state = "SNIR activé" if state == "snir_on" else "SNIR désactivé" if state == "snir_off" else "SNIR inconnu"
        suffix_state = state or "snir_unknown"
        for period in periods:
            filtered = [r for r in state_records if r["packet_interval_s"] == period]
            if not filtered:
                continue
            fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
            if len(clusters) == 1:
                axes = [axes]
            for idx, cluster_id in enumerate(clusters):
                ax = axes[idx]
                for algorithm in algorithms:
                    algo_records = [r for r in filtered if r["algorithm"] == algorithm]
                    algo_records.sort(key=lambda item: item["num_nodes"])
                    xs: List[int] = []
                    ys: List[float] = []
                    for item in algo_records:
                        value = item.get("cluster_pdr", {}).get(cluster_id)
                        if value is None:
                            continue
                        xs.append(item["num_nodes"])
                        ys.append(value)
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
            fig.suptitle(f"PDR par cluster – {label_state} – période {title_period} s")
            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"step1_cluster_pdr_{suffix_state}_tx_{title_period}.png"
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(output, dpi=150)
            plt.close(fig)


def generate_step1_figures(results_dir: Path, figures_dir: Path) -> None:
    if plt is None:
        print("matplotlib n'est pas disponible ; aucune figure générée.")
        return
    records = _load_step1_records(results_dir)
    if not records:
        print(f"Aucun CSV trouvé dans {results_dir} ; rien à tracer.")
        return
    output_dir = figures_dir / "step1"
    _plot_cluster_pdr(records, output_dir)
    _plot_global_metric(records, "PDR", "PDR global", "pdr_global", output_dir)
    _plot_global_metric(records, "DER", "DER global", "der_global", output_dir)
    _plot_global_metric(records, "collisions", "Collisions", "collisions", output_dir)
    _plot_global_metric(records, "jain_index", "Indice de Jain", "jain_index", output_dir)
    _plot_global_metric(records, "throughput_bps", "Débit agrégé (bps)", "throughput", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire contenant les CSV produits par l'étape 1",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire racine de sortie pour les figures",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    generate_step1_figures(args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
