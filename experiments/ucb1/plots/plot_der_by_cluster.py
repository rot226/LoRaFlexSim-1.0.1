"""Trace le DER par cluster pour Step1/QoS et UCB1 avec et sans SNIR.

Deux familles de CSV sont acceptées :
- **Step1/QoS** (résumés ou bruts) avec des colonnes ``cluster_pdr_*`` ou
  ``cluster_der_*`` et un champ indiquant l'état SNIR (``with_snir``,
  ``snir_state``…).
- **UCB1** (``run_ucb1_density_sweep.py``) avec des colonnes ``cluster``,
  ``der``/``pdr`` et éventuellement un indicateur SNIR.

Les courbes sont superposées avec la convention suivante :
- bleu : SNIR désactivé ;
- rouge : SNIR activé ;
- courbes UCB1 en pointillé/losange si présentes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from experiments.ucb1.plots.plot_style import apply_ieee_style, filter_top_groups

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STEP1 = ROOT_DIR / "results" / "step1" / "summary.csv"
DEFAULT_UCB1 = Path(__file__).resolve().parents[1] / "ucb1_density_metrics.csv"
DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "ucb1_baseline_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_der_vs_step1.png"
SNIR_LABELS = {"snir_on": "SNIR activé", "snir_off": "SNIR désactivé", "snir_unknown": "SNIR inconnu"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}
MIXRA_OPT_ALIASES = {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Superpose le DER QoS (Step1) et UCB1 par cluster.")
    parser.add_argument(
        "--step1-csv",
        type=Path,
        default=DEFAULT_STEP1,
        help="CSV Step1/QoS (summary.csv ou brut avec cluster_pdr/der).",
    )
    parser.add_argument(
        "--ucb1-csv",
        type=Path,
        default=DEFAULT_UCB1,
        help="CSV UCB1 issu du balayage de densité.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE,
        help="CSV baseline (MixRA-Opt) issu de run_baseline_comparison.py.",
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
    parser.add_argument(
        "--time-window",
        action="append",
        default=[],
        metavar="DEBUT:FIN",
        help=(
            "Filtre les lignes issues d'agrégats temporels (window_start_s/window_end_s) "
            "sur l'intervalle fourni en secondes. Peut être passé plusieurs fois."
        ),
    )
    parser.add_argument(
        "--window-index",
        action="append",
        type=int,
        default=[],
        help="Sélectionne un ou plusieurs indices de fenêtre (colonne window_index).",
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
    if path:
        if any("snir" in part.lower() for part in path.parts):
            return "snir_on"
    return "snir_unknown"


def _extract_cluster_values(row: Mapping[str, object]) -> Dict[int, float]:
    values: Dict[int, float] = {}
    # Priorité aux moyennes si présentes
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


def _extract_windows(row: Mapping[str, object]) -> dict[str, float | int | None]:
    def _to_float(value: object | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _to_int(value: object | None) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "window_index": _to_int(row.get("window_index")),
        "window_start_s": _to_float(row.get("window_start_s")),
        "window_end_s": _to_float(row.get("window_end_s")),
    }


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
        window_fields = _extract_windows(row)
        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "Step1/QoS",
                "cluster": cluster_id,
                "num_nodes": x_value,
                "der": value,
                "snir_state": snir_state,
                **window_fields,
            })
    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["num_nodes", "der"])


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
        window_fields = _extract_windows(row)
        clusters = _extract_cluster_values(row)
        for cluster_id, value in clusters.items():
            records.append({
                "source": "MixRA-Opt",
                "cluster": cluster_id,
                "num_nodes": x_value,
                "der": value,
                "snir_state": snir_state,
                **window_fields,
            })
    tidy = pd.DataFrame(records)
    return tidy.dropna(subset=["num_nodes", "der"])


def _load_ucb1(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV UCB1 introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = "der" if "der" in df.columns else "pdr" if "pdr" in df.columns else None
    if metric is None:
        raise ValueError(f"Impossible de trouver les colonnes 'der' ou 'pdr' dans {path}")
    required = {"cluster", "num_nodes", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df["source"] = "UCB1"
    df["snir_state"] = df.apply(lambda row: _detect_snir(row, path), axis=1)
    df.rename(columns={metric: "der"}, inplace=True)
    if {"window_start_s", "window_end_s", "window_index"}.issubset(df.columns):
        df[["window_start_s", "window_end_s"]] = df[["window_start_s", "window_end_s"]].apply(pd.to_numeric)
        df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    return df


def _load_baseline(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        print(f"Avertissement : CSV baseline introuvable ({path}), section ignorée.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"cluster", "num_nodes", "der", "algorithm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")

    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        algorithm = _normalize_algorithm(row.get("algorithm"))
        if algorithm != "mixra_opt":
            continue
        try:
            num_nodes = float(row.get("num_nodes"))
            cluster = int(float(row.get("cluster")))
            der = float(row.get("der"))
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "source": "MixRA-Opt",
                "cluster": cluster,
                "num_nodes": num_nodes,
                "der": der,
                "snir_state": "snir_unknown",
            }
        )

    return pd.DataFrame(records)


def _parse_time_windows(raw_windows: Sequence[str]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for raw in raw_windows:
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Fenêtre temporelle invalide '{raw}' (attendu: debut:fin en secondes)")
        start_text, end_text = raw.split(":", maxsplit=1)
        try:
            start = float(start_text)
            end = float(end_text)
        except ValueError as exc:  # pragma: no cover - validation simple
            raise ValueError(
                f"Impossible de parser la fenêtre '{raw}', valeurs numériques attendues"
            ) from exc
        if end <= start:
            raise ValueError(f"Fenêtre temporelle invalide '{raw}' : fin <= début")
        parsed.append((start, end))
    return parsed


def _filter_windows(
    df: pd.DataFrame, time_windows: Sequence[tuple[float, float]], indices: Sequence[int]
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df
    if time_windows and {"window_start_s", "window_end_s"}.issubset(filtered.columns):
        mask = False
        for start, end in time_windows:
            mask |= (filtered["window_start_s"] >= start) & (filtered["window_end_s"] <= end)
        filtered = filtered[mask]
    elif time_windows:
        print(
            "Avertissement : filtres temporels ignorés car les colonnes window_start_s/window_end_s sont absentes."
        )

    if indices and "window_index" in filtered.columns:
        filtered = filtered[filtered["window_index"].isin(indices)]
    elif indices:
        print("Avertissement : filtres par indice de fenêtre ignorés (colonne window_index absente).")

    return filtered


def _plot_der(df: pd.DataFrame, output: Path) -> None:
    df = filter_top_groups(df, ["source", "snir_state", "cluster"], max_groups=3)
    fig, ax = plt.subplots(figsize=(8, 5))
    style_by_source = {
        "Step1/QoS": {"linestyle": "-", "marker": "o"},
        "UCB1": {"linestyle": "--", "marker": "D"},
        "MixRA-Opt": {"linestyle": ":", "marker": "s"},
    }
    for (source, snir_state, cluster_id), subset in df.groupby(["source", "snir_state", "cluster"], sort=True):
        subset = subset.sort_values("num_nodes")
        color = SNIR_COLORS.get(snir_state, SNIR_COLORS["snir_unknown"])
        label = f"{source} C{int(cluster_id)} ({SNIR_LABELS.get(snir_state, snir_state)})"
        style = style_by_source.get(source, {})
        ax.plot(subset["num_nodes"], subset["der"], label=label, color=color, **style)

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("DER / PDR par cluster")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title("DER par cluster")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Figure enregistrée dans {output}")


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    time_windows = _parse_time_windows(args.time_window)
    summary_path = _resolve_summary_path(args.step1_csv)
    baseline_summary = _load_mixra_opt_baseline_from_summary(summary_path)
    baseline_csv = pd.DataFrame() if not baseline_summary.empty else _load_baseline(args.baseline_csv)
    combined = pd.concat(
        [
            _load_step1(args.step1_csv),
            _load_ucb1(args.ucb1_csv),
            baseline_summary,
            baseline_csv,
        ],
        ignore_index=True,
    )
    combined = _filter_network_sizes(combined, args.network_sizes)
    combined = _filter_windows(combined, time_windows, args.window_index)
    if combined.empty:
        raise SystemExit("Aucune donnée exploitable pour tracer la figure.")
    _plot_der(combined, args.output)


if __name__ == "__main__":
    main()
