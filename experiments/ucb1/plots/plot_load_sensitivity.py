"""Compare la sensibilité au trafic pour Step1/QoS et UCB1 (SNIR on/off).

Les scripts Step1/QoS génèrent des colonnes ``cluster_pdr_*``/``cluster_der_*``
sur plusieurs intervalles d'émission. Les CSV UCB1 (``run_ucb1_load_sweep.py``)
exposent des colonnes ``cluster`` et ``der``/``pdr``. Les deux sources peuvent
inclure un indicateur SNIR ; à défaut il est déduit du chemin ou des flags.

Couleurs : bleu (SNIR désactivé) et rouge (SNIR activé). Les courbes UCB1 sont
tracées en pointillé avec des marqueurs losange pour être différenciées.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout, save_figure
from plot_defaults import resolve_ieee_figsize
from experiments.ucb1.plots.plot_style import apply_plot_style, filter_top_groups

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STEP1 = ROOT_DIR / "results" / "step1" / "summary.csv"
DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_load_vs_step1.png"
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé", "snir_unknown": "SNIR inconnu"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}
MIXRA_OPT_ALIASES = {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Superpose le DER par cluster vs charge (Step1 + UCB1).")
    parser.add_argument("--step1-csv", type=Path, default=DEFAULT_STEP1, help="CSV Step1/QoS (summary ou brut).")
    parser.add_argument("--ucb1-csv", type=Path, default=DEFAULT_UCB1, help="CSV UCB1 du balayage de charge.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin du PNG à générer (répertoires créés si besoin).",
    )
    return parser.parse_args()


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
        x_value = row.get("packet_interval_s")
        if x_value is None:
            x_value = row.get("packet_interval")
        try:
            x_value = float(x_value)
        except (TypeError, ValueError):
            x_value = None
        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "Step1/QoS",
                "cluster": cluster_id,
                "packet_interval": x_value,
                "der": value,
                "snir_state": snir_state,
            })
    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["packet_interval", "der"])


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
        x_value = row.get("packet_interval_s")
        if x_value is None:
            x_value = row.get("packet_interval")
        try:
            x_value = float(x_value)
        except (TypeError, ValueError):
            x_value = None
        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "MixRA-Opt",
                "cluster": cluster_id,
                "packet_interval": x_value,
                "der": value,
                "snir_state": snir_state,
            })
    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["packet_interval", "der"])


def _load_ucb1(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV UCB1 introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = "der" if "der" in df.columns else "pdr" if "pdr" in df.columns else None
    if metric is None:
        raise ValueError(f"Impossible de trouver les colonnes 'der' ou 'pdr' dans {path}")
    interval_col = "packet_interval" if "packet_interval" in df.columns else "packet_interval_s"
    required = {"cluster", metric, interval_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df.rename(columns={metric: "der", interval_col: "packet_interval"}, inplace=True)
    df["source"] = "UCB1"
    df["snir_state"] = df.apply(lambda row: _detect_snir(row, path), axis=1)
    return df


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
        subset = subset.sort_values("packet_interval")
        color = SNIR_COLORS.get(snir_state, SNIR_COLORS["snir_unknown"])
        label = f"{source} C{int(cluster_id)} ({SNIR_LABELS.get(snir_state, snir_state)})"
        style = style_by_source.get(source, {})
        ax.plot(subset["packet_interval"] / 60.0, subset["der"], label=label, color=color, **style)

    ax.set_xlabel("Intervalle entre paquets (minutes)")
    ax.set_ylabel("DER / PDR par cluster")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title("Sensibilité à la charge")

    apply_figure_layout(fig, tight_layout=True)
    save_figure(fig, output.parent, output.stem)
    print(f"Figure enregistrée dans {output.parent / f'{output.stem}.png'}")


def main() -> None:
    apply_plot_style()
    args = parse_args()
    summary_path = _resolve_summary_path(args.step1_csv)
    baseline_summary = _load_mixra_opt_baseline_from_summary(summary_path)
    combined = pd.concat(
        [
            _load_step1(args.step1_csv),
            _load_ucb1(args.ucb1_csv),
            baseline_summary,
        ],
        ignore_index=True,
    )
    if combined.empty:
        raise SystemExit("Aucune donnée exploitable pour tracer la figure.")
    _plot(combined, args.output)


if __name__ == "__main__":
    main()
