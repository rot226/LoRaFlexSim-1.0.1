"""Superpose un débit global et par cluster pour Step1/QoS et UCB1.

Les CSV Step1/QoS peuvent fournir des valeurs globales (``DER``/``DER_mean``)
et/ou des colonnes ``cluster_pdr_*``. Les CSV UCB1 utilisent les colonnes
``cluster`` et ``der``/``pdr``. Le même code détecte l'état SNIR (bleu = off,
rouge = on) et ajoute des courbes pointillées pour UCB1.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout, save_figure
from plot_defaults import resolve_ieee_figsize
from experiments.ucb1.plots.plot_style import apply_plot_style, filter_top_groups

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STEP1 = ROOT_DIR / "results" / "step1" / "summary.csv"
DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_density_metrics.csv"
DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "ucb1_baseline_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_throughput_vs_step1.png"
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé", "snir_unknown": "SNIR inconnu"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}
MIXRA_OPT_ALIASES = {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Débit global et par cluster – Step1/QoS vs UCB1.")
    parser.add_argument("--step1-csv", type=Path, default=DEFAULT_STEP1, help="CSV Step1/QoS (summary ou brut).")
    parser.add_argument("--ucb1-csv", type=Path, default=DEFAULT_UCB1, help="CSV UCB1 du balayage de densité.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE,
        help="CSV baseline (MixRA-Opt) issu de run_baseline_comparison.py.",
    )
    parser.add_argument(
        "--metric",
        choices=["success_rate", "der", "pdr"],
        default="der",
        help="Colonne utilisée comme indicateur de débit pour UCB1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin du PNG à générer (répertoires créés si besoin).",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    return parser.parse_args()


def _filter_network_sizes(df: pd.DataFrame, network_sizes: List[int] | None) -> pd.DataFrame:
    if not network_sizes or "num_nodes" not in df.columns:
        return df
    available = sorted(df["num_nodes"].dropna().unique())
    requested = sorted({int(size) for size in network_sizes})
    missing = sorted(set(requested) - {int(value) for value in available})
    if missing:
        warnings.warn(
            "Tailles de réseau demandées absentes: "
            + ", ".join(str(size) for size in missing),
            stacklevel=2,
        )
    return df[df["num_nodes"].isin(requested)]


def _parse_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _normalize_algorithm(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if normalized in MIXRA_OPT_ALIASES:
        return "mixra_opt"
    return text


def _detect_snir(row: Mapping[str, object], path: Path | None) -> str:
    candidates = [row.get("with_snir"), row.get("use_snir"), row.get("snir_enabled"), row.get("snir")]
    for candidate in candidates:
        parsed = _parse_bool(candidate)
        if parsed is not None:
            return "snir_on" if parsed else "snir_off"
    state = row.get("snir_state")
    if isinstance(state, str) and state:
        state_lower = state.lower()
        if "on" in state_lower:
            return "snir_on"
        if "off" in state_lower:
            return "snir_off"
    if path and any("snir" in part.lower() for part in path.parts):
        return "snir_on"
    return "snir_unknown"


def _extract_cluster_values(row: Mapping[str, object]) -> Dict[int, float]:
    values: Dict[int, float] = {}
    mean_keys: MutableMapping[int, float] = {}
    direct_keys: MutableMapping[int, float] = {}
    pattern = re.compile(r"cluster_(?:pdr|der)[^0-9]*([0-9]+)")
    for key, raw_value in row.items():
        match = pattern.match(str(key))
        if not match:
            continue
        try:
            cid = int(match.group(1))
        except ValueError:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if "mean" in str(key):
            mean_keys[cid] = value
        elif cid not in direct_keys:
            direct_keys[cid] = value
    values.update(direct_keys)
    values.update(mean_keys)
    return values


def _resolve_summary_path(step1_path: Path) -> Path | None:
    if step1_path and step1_path.name == "summary.csv" and step1_path.exists():
        return step1_path
    candidate = step1_path.parent / "summary.csv"
    if candidate.exists():
        return candidate
    return None


def _load_step1(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV Step1 introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    records: List[dict] = []
    for row in df.to_dict(orient="records"):
        snir_state = _detect_snir(row, path)
        x_value = row.get("num_nodes")
        try:
            x_value = float(x_value)
        except (TypeError, ValueError):
            x_value = None

        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "Step1/QoS",
                "cluster": cluster_id,
                "num_nodes": x_value,
                "value": value,
                "snir_state": snir_state,
            })

        for global_key in ("DER_mean", "DER", "PDR", "PDR_mean"):
            if global_key in row:
                try:
                    global_value = float(row[global_key])
                except (TypeError, ValueError):
                    continue
                records.append({
                    "source": "Step1/QoS",
                    "cluster": "Global",
                    "num_nodes": x_value,
                    "value": global_value,
                    "snir_state": snir_state,
                })
                break

    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["num_nodes", "value"])


def _load_mixra_opt_baseline_from_summary(summary_path: Path | None) -> pd.DataFrame:
    if not summary_path or not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    if "algorithm" not in df.columns:
        print(f"Avertissement : summary.csv sans colonne algorithm ({summary_path}).")
        return pd.DataFrame()
    df = df.copy()
    df["algorithm_norm"] = df["algorithm"].apply(_normalize_algorithm)
    df = df[df["algorithm_norm"] == "mixra_opt"]
    if df.empty:
        return pd.DataFrame()
    records: List[dict] = []
    for row in df.to_dict(orient="records"):
        snir_state = _detect_snir(row, summary_path)
        try:
            x_value = float(row.get("num_nodes"))
        except (TypeError, ValueError):
            x_value = None
        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "MixRA-Opt",
                "cluster": cluster_id,
                "num_nodes": x_value,
                "value": value,
                "snir_state": snir_state,
            })

        for global_key in ("DER_mean", "DER", "PDR", "PDR_mean"):
            if global_key in row:
                try:
                    global_value = float(row[global_key])
                except (TypeError, ValueError):
                    continue
                records.append({
                    "source": "MixRA-Opt",
                    "cluster": "Global",
                    "num_nodes": x_value,
                    "value": global_value,
                    "snir_state": snir_state,
                })
                break

    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["num_nodes", "value"])


def _load_ucb1(path: Path, metric: str) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV UCB1 introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = metric if metric in df.columns else next((col for col in ("der", "pdr", "success_rate") if col in df.columns), None)
    if metric is None:
        raise ValueError(f"Impossible de trouver les colonnes de débit dans {path}")
    required = {"cluster", "num_nodes", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df.rename(columns={metric: "value"}, inplace=True)
    df["source"] = "UCB1"
    df["snir_state"] = df.apply(lambda row: _detect_snir(row, path), axis=1)

    # Ajoute une moyenne simple par num_nodes / état SNIR
    global_rows: List[dict] = []
    for (num_nodes, snir_state), group in df.groupby(["num_nodes", "snir_state"], sort=False):
        global_rows.append(
            {
                "source": "UCB1",
                "cluster": "Global",
                "num_nodes": num_nodes,
                "value": float(group["value"].mean()),
                "snir_state": snir_state,
            }
        )
    global_df = pd.DataFrame(global_rows)
    return pd.concat([df, global_df], ignore_index=True)


def _load_baseline(path: Path, metric: str) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV baseline introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = metric if metric in df.columns else next((col for col in ("der", "pdr", "success_rate") if col in df.columns), None)
    if metric is None:
        raise ValueError(f"Impossible de trouver les colonnes de débit dans {path}")
    required = {"cluster", "num_nodes", metric, "algorithm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    records: List[dict] = []
    for row in df.to_dict(orient="records"):
        algorithm = _normalize_algorithm(row.get("algorithm"))
        if algorithm != "mixra_opt":
            continue
        try:
            num_nodes = float(row.get("num_nodes"))
            cluster = int(float(row.get("cluster")))
            value = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "source": "MixRA-Opt",
                "cluster": cluster,
                "num_nodes": num_nodes,
                "value": value,
                "snir_state": "snir_unknown",
            }
        )

    if not records:
        return pd.DataFrame()
    baseline_df = pd.DataFrame(records)
    global_rows: List[dict] = []
    for (num_nodes, snir_state), group in baseline_df.groupby(["num_nodes", "snir_state"], sort=False):
        global_rows.append(
            {
                "source": "MixRA-Opt",
                "cluster": "Global",
                "num_nodes": num_nodes,
                "value": float(group["value"].mean()),
                "snir_state": snir_state,
            }
        )
    global_df = pd.DataFrame(global_rows)
    return pd.concat([baseline_df, global_df], ignore_index=True)


def _plot(df: pd.DataFrame, output: Path) -> None:
    df = filter_top_groups(df, ["source", "snir_state", "cluster"], max_groups=3)
    series_count = max(1, df["source"].nunique())
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    style_by_source = {
        "Step1/QoS": {"linestyle": "-", "marker": "o"},
        "UCB1": {"linestyle": "--", "marker": "D"},
        "MixRA-Opt": {"linestyle": ":", "marker": "s"},
    }
    for (source, snir_state, cluster_id), subset in df.groupby(["source", "snir_state", "cluster"], sort=True):
        subset = subset.sort_values("num_nodes")
        color = SNIR_COLORS.get(snir_state, SNIR_COLORS["snir_unknown"])
        label_cluster = "Global" if str(cluster_id).lower() == "global" else f"C{int(cluster_id)}"
        label = f"{source} {label_cluster} ({SNIR_LABELS.get(snir_state, snir_state)})"
        style = style_by_source.get(source, {})
        ax.plot(subset["num_nodes"], subset["value"], label=label, color=color, **style)

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("Taux de succès / débit relatif")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title("Débit global vs cluster")

    apply_figure_layout(fig, tight_layout=True)
    save_figure(fig, output.parent, output.stem)
    print(f"Figure enregistrée dans {output.parent / f'{output.stem}.png'}")


def main() -> None:
    apply_plot_style()
    args = parse_args()
    summary_path = _resolve_summary_path(args.step1_csv)
    baseline_summary = _load_mixra_opt_baseline_from_summary(summary_path)
    baseline_csv = pd.DataFrame() if not baseline_summary.empty else _load_baseline(args.baseline_csv, args.metric)
    combined = pd.concat([
        _load_step1(args.step1_csv),
        _load_ucb1(args.ucb1_csv, args.metric),
        baseline_summary,
        baseline_csv,
    ], ignore_index=True)
    combined = _filter_network_sizes(combined, args.network_sizes)
    if combined.empty:
        raise SystemExit("Aucune donnée exploitable pour tracer la figure.")
    _plot(combined, args.output)


if __name__ == "__main__":
    main()
