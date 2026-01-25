"""CLI pour générer des visualisations QoS à partir des résultats LoRaFlexSim."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:  # pragma: no cover - dépend du mode d'exécution
    from .lfs_metrics import (
        MethodScenarioMetrics,
        load_all_metrics,
        load_gateway_position,
        load_yaml_config,
    )
except ImportError:  # pragma: no cover - fallback pour exécution directe
    from lfs_metrics import (
        MethodScenarioMetrics,
        load_all_metrics,
        load_gateway_position,
        load_yaml_config,
    )


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "*",
    "X",
    "<",
    ">",
]

SNIR_STATE_COLORS = {
    "snir_on": "#d62728",  # rouge
    "snir_off": "#1f77b4",  # bleu
    "snir_unknown": "#7f7f7f",
}

SNIR_STATE_LABELS = {
    "snir_on": "SNIR activé",
    "snir_off": "SNIR désactivé",
    "snir_unknown": "SNIR inconnu",
}

MAX_CURVES_PER_PANEL = 4


def _method_groups(
    methods: Sequence[str],
    per_algo: bool,
    max_per_panel: int = MAX_CURVES_PER_PANEL,
) -> List[List[str]]:
    if not per_algo or max_per_panel <= 0:
        return [list(methods)]
    if len(methods) <= max_per_panel:
        return [list(methods)]
    return [list(methods[i : i + max_per_panel]) for i in range(0, len(methods), max_per_panel)]


def _panel_title(base: str, methods: Sequence[str]) -> str:
    if not methods:
        return base
    if base:
        return f"{base}\nMéthodes: {', '.join(methods)}"
    return f"Méthodes: {', '.join(methods)}"


def _render_snir_variants(
    render: Callable[[List[str], str, str], Optional[Path]],
    *,
    on_title: str,
    off_title: str,
    mixed_title: str,
) -> List[Path]:
    """Rend systématiquement les variantes SNIR ON/OFF/mixte via un callback.

    Le callback ``render`` reçoit la liste des états à tracer, un suffixe de
    fichier (incluant le soulignement) et le titre associé, puis retourne
    éventuellement le chemin sauvegardé. La fonction garantit la production des
    trois combinaisons utilisées par la CLI (_snir-on, _snir-off, _snir-mixed).
    """

    variants: List[Tuple[List[str], str, str]] = [
        (["snir_on"], "_snir-on", on_title),
        (["snir_off"], "_snir-off", off_title),
        (["snir_on", "snir_off", "snir_unknown"], "_snir-mixed", mixed_title),
    ]

    saved: List[Path] = []
    for states, suffix, title in variants:
        path = render(states, suffix, title)
        if path is not None:
            saved.append(path)
    return saved


def _style_mapping(labels: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        mapping[str(label)] = (color, marker)
    return mapping


def _values_for_attribute(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    attribute: str,
) -> Dict[str, List[float]]:
    values_per_method: Dict[str, List[float]] = {}
    for method, scenario_metrics in metrics_by_method.items():
        values: List[float] = []
        for scenario in scenarios:
            metric = scenario_metrics.get(scenario)
            value = getattr(metric, attribute) if metric else None
            if value is None:
                values.append(float("nan"))
            else:
                values.append(float(value))
        values_per_method[method] = values
    return values_per_method


def _metric_snir_state(metric: Optional[MethodScenarioMetrics]) -> str:
    if metric is None:
        return "snir_unknown"
    if metric.use_snir is True:
        return "snir_on"
    if metric.use_snir is False:
        return "snir_off"
    if metric.snir_state:
        normalized = metric.snir_state.strip().lower()
        if normalized in SNIR_STATE_LABELS:
            return normalized
        if normalized in {"snir-on", "on", "enabled"}:
            return "snir_on"
        if normalized in {"snir-off", "off", "disabled"}:
            return "snir_off"
    return "snir_unknown"


def _values_by_snir_state(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    attribute: str,
) -> Dict[str, Dict[str, List[float]]]:
    states = {"snir_on", "snir_off", "snir_unknown"}
    values_per_state: Dict[str, Dict[str, List[float]]] = {
        state: {method: [float("nan")] * len(scenarios) for method in metrics_by_method}
        for state in states
    }

    for method, scenario_metrics in metrics_by_method.items():
        for idx, scenario in enumerate(scenarios):
            metric = scenario_metrics.get(scenario)
            if metric is None:
                continue
            value = getattr(metric, attribute, None)
            if value is None:
                continue
            state = _metric_snir_state(metric)
            if state not in values_per_state:
                continue
            values_per_state[state][method][idx] = float(value)
    return values_per_state


def _all_nan(values: Sequence[float]) -> bool:
    return all(math.isnan(value) for value in values)


def _ratio_confidence(successes: int, attempts: int) -> Tuple[float, float]:
    if attempts <= 0:
        return float("nan"), float("nan")
    p_hat = successes / attempts
    margin = 1.96 * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / attempts))
    return max(0.0, p_hat - margin), min(1.0, p_hat + margin)


def _ratio_ci_for_metric(metric: MethodScenarioMetrics) -> Tuple[float, float]:
    return _ratio_confidence(metric.delivered, metric.attempted)


def _snir_ci_for_metric(metric: MethodScenarioMetrics) -> Tuple[float, float]:
    if metric.snir_ci_low is None or metric.snir_ci_high is None:
        return float("nan"), float("nan")
    return float(metric.snir_ci_low), float(metric.snir_ci_high)


def _moving_average(values: Sequence[float], window_size: int) -> List[float]:
    if window_size <= 1:
        return [float(value) for value in values]
    series = pd.Series(values, dtype=float)
    return series.rolling(window=window_size, min_periods=1).mean().tolist()


def _rolling_metrics(
    df: pd.DataFrame, window_size: float, window_mode: str
) -> pd.DataFrame:
    if df.empty:
        return df

    df_sorted = df.sort_values("x").reset_index(drop=True)
    times = df_sorted["x"].to_numpy(dtype=float)
    delivered = df_sorted["delivered"].to_numpy(dtype=float)
    snir = df_sorted["snir"].to_numpy(dtype=float)

    results: List[dict] = []
    cumulative_delivered = np.cumsum(delivered)

    def _window_start_for_index(index: int) -> int:
        if window_mode == "packets":
            width = max(1, int(math.ceil(window_size)))
            return max(0, index - width + 1)
        start = 0
        while start <= index and times[index] - times[start] > window_size:
            start += 1
        return start

    for idx in range(len(times)):
        start = _window_start_for_index(idx)
        attempts = idx - start + 1
        delivered_count = cumulative_delivered[idx] - (cumulative_delivered[start - 1] if start > 0 else 0)
        pdr_value = delivered_count / attempts if attempts > 0 else float("nan")
        ci_low, ci_high = _ratio_confidence(int(delivered_count), attempts)

        window_snir = snir[start : idx + 1]
        valid_snir = window_snir[~np.isnan(window_snir)]
        if valid_snir.size > 0:
            snir_mean = float(np.mean(valid_snir))
            snir_error = 0.0
            if valid_snir.size > 1:
                snir_error = 1.96 * float(np.std(valid_snir, ddof=1) / math.sqrt(valid_snir.size))
            snir_low = snir_mean - snir_error
            snir_high = snir_mean + snir_error
        else:
            snir_mean = snir_low = snir_high = float("nan")

        results.append(
            {
                "x": times[idx],
                "attempts": attempts,
                "pdr": pdr_value,
                "pdr_low": ci_low,
                "pdr_high": ci_high,
                "der": pdr_value,
                "der_low": ci_low,
                "der_high": ci_high,
                "snir": snir_mean,
                "snir_low": snir_low,
                "snir_high": snir_high,
            }
        )

    return pd.DataFrame(results)


def _load_packet_timeseries(
    root: Path, method: str, scenario: str, *, require_time: bool
) -> Optional[pd.DataFrame]:
    path = root / method / scenario / "packets.csv"
    if not path.is_file():
        return None

    df = pd.read_csv(path)
    success_series = _extract_success_series(df)
    if success_series is None:
        return None
    time_series = _extract_time_series(df)
    if require_time and time_series is None:
        raise RuntimeError(
            "Impossible d'aligner la fenêtre temporelle : aucune colonne de temps détectée dans "
            f"{path}"
        )
    snir_series = _extract_snir_series(df)

    timeline = time_series if time_series is not None else pd.Series(range(len(success_series)), dtype=float)

    prepared = pd.DataFrame(
        {
            "x": pd.to_numeric(timeline, errors="coerce"),
            "delivered": pd.to_numeric(success_series, errors="coerce"),
            "snir": pd.to_numeric(snir_series, errors="coerce") if snir_series is not None else np.nan,
        }
    )
    prepared = prepared.dropna(subset=["x", "delivered"]).reset_index(drop=True)
    return prepared


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit l'espace de noms d'arguments pour la CLI de génération de graphiques."""

    parser = argparse.ArgumentParser(
        prog="lfs_plots",
        description=(
            "Génère des figures (PDR global/cluster, DER, collisions, énergie, équité, part SF et CDF SNIR) à partir des résultats QoS."
        ),
    )
    parser.add_argument(
        "--in",
        dest="root",
        type=Path,
        required=True,
        help="Dossier racine contenant les résultats agrégés (<méthode>/<scénario>).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        required=False,
        help="Fichier YAML décrivant les scénarios (utilisé pour ordonner les sorties).",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=Path("qos_cli") / "figures",
        help="Dossier de destination des figures (défaut : qos_cli/figures).",
    )
    parser.add_argument(
        "--rolling-window",
        dest="rolling_window",
        type=float,
        default=200.0,
        help=(
            "Taille de la fenêtre glissante pour les métriques temporelles. "
            "Interprétée comme un nombre de paquets si --window-mode=packets, ou comme "
            "une durée en secondes si --window-mode=duration."
        ),
    )
    parser.add_argument(
        "--window-mode",
        dest="window_mode",
        choices=["packets", "duration"],
        default="packets",
        help="Définit l'alignement de la fenêtre glissante : par nombre de paquets ou par durée (s).",
    )
    parser.add_argument(
        "--moving-average-window",
        dest="moving_average_window",
        type=int,
        default=3,
        help="Taille de la fenêtre de moyenne mobile (par nombre de scénarios).",
    )
    parser.add_argument(
        "--per-algo",
        dest="per_algo",
        action="store_true",
        help="Découpe les figures en panneaux avec 4 algorithmes maximum par panneau.",
    )
    return parser.parse_args(argv)


def _extract_success_series(df: pd.DataFrame) -> Optional[pd.Series]:
    success_columns = [
        "delivered",
        "success",
        "is_delivered",
        "rx_success",
        "successful",
    ]
    for column in success_columns:
        if column in df.columns:
            series = df[column]
            if series.dtype == bool:
                return series.astype(int)
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors="coerce")
            return series.astype(str).str.lower().isin({"true", "1", "yes", "delivered", "success"}).astype(int)
    return None


def _extract_snir_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in ["snir_dB", "snir_db", "snir"]:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                return series
    return None


def _extract_time_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in [
        "time",
        "timestamp",
        "sent_time",
        "tx_time",
        "emission_time",
        "sim_time",
        "event_time",
    ]:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                return series
    return None


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _load_nodes_df(root: Path, method: str, scenario: str) -> Optional[pd.DataFrame]:
    path = root / method / scenario / "nodes.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _load_packets_df(root: Path, method: str, scenario: str) -> Optional[pd.DataFrame]:
    path = root / method / scenario / "packets.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _extract_node_count(nodes_df: Optional[pd.DataFrame]) -> Optional[int]:
    if nodes_df is None or nodes_df.empty:
        return None
    node_column = _find_column(
        list(nodes_df.columns),
        ["node_id", "device", "device_id", "end_device", "devaddr"],
    )
    if node_column is None:
        return int(len(nodes_df.index))
    return int(nodes_df[node_column].nunique())


def _extract_mean_sf(nodes_df: Optional[pd.DataFrame]) -> Optional[float]:
    if nodes_df is None or nodes_df.empty:
        return None
    sf_column = _find_column(
        list(nodes_df.columns),
        ["sf", "spreading_factor", "SF", "assigned_sf"],
    )
    if sf_column is None:
        return None
    sf_values = pd.to_numeric(nodes_df[sf_column], errors="coerce").dropna()
    if sf_values.empty:
        return None
    return float(sf_values.mean())


def _scenario_nodes_from_config(
    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]],
    scenario: str,
) -> Optional[int]:
    if scenarios_cfg is None:
        return None
    scenario_cfg = scenarios_cfg.get(scenario)
    if not isinstance(scenario_cfg, Mapping):
        return None
    for key in ["N", "n", "nodes", "node_count", "nb_nodes"]:
        if key not in scenario_cfg:
            continue
        try:
            return int(float(scenario_cfg[key]))
        except (TypeError, ValueError):
            continue
    return None


def _scenario_nodes_from_name(scenario: str) -> Optional[int]:
    match = re.search(r"N(\d+)", scenario)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _scenario_node_counts(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    metrics_root: Path,
    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]],
) -> Dict[str, Optional[int]]:
    counts: Dict[str, Optional[int]] = {}
    methods = sorted(metrics_by_method.keys())
    for scenario in scenarios:
        count = _scenario_nodes_from_config(scenarios_cfg, scenario)
        if count is None:
            for method in methods:
                nodes_df = _load_nodes_df(metrics_root, method, scenario)
                count = _extract_node_count(nodes_df)
                if count is not None:
                    break
        if count is None:
            count = _scenario_nodes_from_name(scenario)
        counts[scenario] = count
    return counts


def _ordered_scenarios_by_nodes(
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
) -> List[str]:
    sortable: List[Tuple[int, str]] = []
    for scenario in scenarios:
        count = node_counts.get(scenario)
        if count is None:
            continue
        sortable.append((count, scenario))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [scenario for _, scenario in sortable]


def build_method_mapping(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics]
) -> Dict[str, Dict[str, MethodScenarioMetrics]]:
    """Réorganise les métriques par méthode puis par scénario."""

    grouped: Dict[str, Dict[str, MethodScenarioMetrics]] = {}
    for (method, scenario), data in metrics.items():
        grouped.setdefault(method, {})[scenario] = data
    return grouped


def ordered_scenarios(
    metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
    config: Optional[Mapping[str, Mapping[str, object]]],
) -> List[str]:
    """Retourne la liste ordonnée des scénarios détectés."""

    if config:
        order = [str(key) for key in config.keys()]
    else:
        order = sorted({scenario for _, scenario in metrics.keys()})
    available = {scenario for _, scenario in metrics.keys()}
    return [scenario for scenario in order if scenario in available]


def plot_cluster_pdr(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    """Trace le PDR par cluster pour chaque méthode en fonction du scénario."""

    if not scenarios or not metrics_by_method:
        return None

    methods = sorted(metrics_by_method.keys())
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), max(3.5, 2.5 * len(methods))),
    )
    if len(methods) == 1:
        axes = [axes]  # type: ignore[assignment]

    x_positions = list(range(len(scenarios)))

    method_styles = _style_mapping(methods)

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        clusters = sorted(
            {
                cluster
                for metric in method_metrics.values()
                for cluster in metric.cluster_pdr.keys()
            }
        )
        cluster_styles = _style_mapping(clusters)
        if not clusters:
            ax.text(
                0.5,
                0.5,
                "No cluster PDR data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.0)
            continue
        for cluster in clusters:
            values: List[float] = []
            for scenario in scenarios:
                metric = method_metrics.get(scenario)
                values.append(metric.cluster_pdr.get(cluster, float("nan")) if metric else float("nan"))
            if _all_nan(values):
                continue
            color, marker = cluster_styles[str(cluster)]
            ax.plot(
                x_positions,
                values,
                marker=marker,
                color=color,
                label=str(cluster),
            )
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.text(
                0.5,
                0.5,
                "Clusters unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    for ax, method in zip(axes, methods):
        color, _ = method_styles[method]
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.2)

    axes[-1].set_xticks(x_positions, scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "pdr_clusters_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_pdr_vs_nodes(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None
    x_values = [node_counts[scenario] for scenario in ordered]
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(7.0, 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = []
            for scenario in ordered:
                metric = metrics_by_method.get(method, {}).get(scenario)
                value = metric.pdr_global if metric else None
                values.append(float(value) if value is not None else float("nan"))
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_values, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_ylabel("PDR global")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "PDR indisponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "pdr_global_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_der_vs_nodes(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None
    x_values = [node_counts[scenario] for scenario in ordered]
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(7.0, 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = []
            for scenario in ordered:
                metric = metrics_by_method.get(method, {}).get(scenario)
                value = metric.der_global if metric else None
                values.append(float(value) if value is not None else float("nan"))
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_values, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_ylabel("DER global")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "DER indisponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "der_global_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_cluster_pdr_vs_nodes(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None

    methods = sorted(metrics_by_method.keys())
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        sharex=True,
        figsize=(7.0, max(3.5, 2.5 * len(methods))),
    )
    if len(methods) == 1:
        axes = [axes]  # type: ignore[assignment]

    x_values = [node_counts[scenario] for scenario in ordered]
    method_styles = _style_mapping(methods)

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        clusters = sorted(
            {
                cluster
                for metric in method_metrics.values()
                for cluster in metric.cluster_pdr.keys()
            }
        )
        cluster_styles = _style_mapping(clusters)
        if not clusters:
            ax.text(
                0.5,
                0.5,
                "No cluster PDR data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.0)
            continue
        for cluster in clusters:
            values: List[float] = []
            for scenario in ordered:
                metric = method_metrics.get(scenario)
                values.append(metric.cluster_pdr.get(cluster, float("nan")) if metric else float("nan"))
            if _all_nan(values):
                continue
            color, marker = cluster_styles[str(cluster)]
            ax.plot(
                x_values,
                values,
                marker=marker,
                color=color,
                label=str(cluster),
            )
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.text(
                0.5,
                0.5,
                "Clusters unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    for ax, method in zip(axes, methods):
        color, _ = method_styles[method]
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.2)

    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "pdr_clusters_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_snir_mean_vs_nodes_by_sf(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None

    methods = sorted(metrics_by_method.keys())
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        sharex=True,
        figsize=(7.0, max(3.5, 2.5 * len(methods))),
    )
    if len(methods) == 1:
        axes = [axes]  # type: ignore[assignment]

    x_values = [node_counts[scenario] for scenario in ordered]
    method_styles = _style_mapping(methods)

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        sfs = sorted(
            {
                sf
                for metric in method_metrics.values()
                for sf in metric.snir_mean_by_sf.keys()
            },
            key=lambda value: float(value) if str(value).replace(".", "").isdigit() else str(value),
        )
        sf_styles = _style_mapping(sfs)
        if not sfs:
            ax.text(
                0.5,
                0.5,
                "No SNIR/SF data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("SNIR (dB)")
            continue
        for sf in sfs:
            values: List[float] = []
            for scenario in ordered:
                metric = method_metrics.get(scenario)
                value = None
                if metric:
                    value = metric.snir_mean_by_sf.get(sf, {}).get("mean")
                values.append(float(value) if value is not None else float("nan"))
            if _all_nan(values):
                continue
            color, marker = sf_styles[str(sf)]
            ax.plot(
                x_values,
                values,
                marker=marker,
                color=color,
                label=f"SF {sf}",
            )
        ax.set_ylabel("SNIR moyen (dB)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
    for ax, method in zip(axes, methods):
        color, _ = method_styles[method]
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.2)

    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "snir_mean_by_sf_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_snir_mean_vs_nodes_by_cluster(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None

    methods = sorted(metrics_by_method.keys())
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        sharex=True,
        figsize=(7.0, max(3.5, 2.5 * len(methods))),
    )
    if len(methods) == 1:
        axes = [axes]  # type: ignore[assignment]

    x_values = [node_counts[scenario] for scenario in ordered]
    method_styles = _style_mapping(methods)

    for ax, method in zip(axes, methods):
        method_metrics = metrics_by_method.get(method, {})
        clusters = sorted(
            {
                cluster
                for metric in method_metrics.values()
                for cluster in metric.snir_mean_by_cluster.keys()
            }
        )
        cluster_styles = _style_mapping(clusters)
        if not clusters:
            ax.text(
                0.5,
                0.5,
                "No SNIR/cluster data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("SNIR (dB)")
            continue
        for cluster in clusters:
            values: List[float] = []
            for scenario in ordered:
                metric = method_metrics.get(scenario)
                value = None
                if metric:
                    value = metric.snir_mean_by_cluster.get(cluster)
                values.append(float(value) if value is not None else float("nan"))
            if _all_nan(values):
                continue
            color, marker = cluster_styles[str(cluster)]
            ax.plot(
                x_values,
                values,
                marker=marker,
                color=color,
                label=str(cluster),
            )
        ax.set_ylabel("SNIR moyen (dB)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
    for ax, method in zip(axes, methods):
        color, _ = method_styles[method]
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.2)

    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "snir_mean_by_cluster_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metric_with_snir_states(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    attribute: str,
    ylabel: str,
    filename_base: str,
    y_limits: Tuple[Optional[float], Optional[float]] | None = (0.0, 1.0),
    ci_resolver: Optional[Callable[[MethodScenarioMetrics], Tuple[float, float]]] = None,
    per_algo: bool = False,
) -> List[Path]:
    if not scenarios or not metrics_by_method:
        return []

    x_positions = list(range(len(scenarios)))
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    values_by_state = _values_by_snir_state(metrics_by_method, scenarios, attribute)

    def render(states_to_plot: List[str], suffix: str, title: str) -> Optional[Path]:
        method_groups = _method_groups(methods, per_algo)
        fig, axes = plt.subplots(
            nrows=len(method_groups),
            ncols=1,
            sharex=True,
            figsize=(max(6.0, 2.5 * len(scenarios)), 4.5 * len(method_groups)),
        )
        if len(method_groups) == 1:
            axes = [axes]  # type: ignore[assignment]
        for ax, group in zip(axes, method_groups):
            plotted = False
            for state in states_to_plot:
                state_values = values_by_state.get(state, {})
                color = SNIR_STATE_COLORS.get(state, "#7f7f7f")
                label_state = SNIR_STATE_LABELS.get(state, state)
                for method in group:
                    values = state_values.get(method, [])
                    if not values or _all_nan(values):
                        continue
                    _, marker = method_styles[method]
                    ax.plot(
                        x_positions,
                        values,
                        marker=marker,
                        color=color,
                        label=f"{method} ({label_state})",
                    )
                    if ci_resolver is not None:
                        lower: List[float] = []
                        upper: List[float] = []
                        method_metrics = metrics_by_method.get(method, {})
                        for scenario in scenarios:
                            metric = method_metrics.get(scenario)
                            if metric is None or _metric_snir_state(metric) != state:
                                lower.append(float("nan"))
                                upper.append(float("nan"))
                                continue
                            ci_low, ci_high = ci_resolver(metric)
                            lower.append(float(ci_low))
                            upper.append(float(ci_high))
                        if not _all_nan(lower) and not _all_nan(upper):
                            ax.fill_between(x_positions, lower, upper, color=color, alpha=0.15)
                    plotted = True

            ax.set_xticks(x_positions)
            ax.set_ylabel(ylabel)
            if y_limits is not None:
                lower, upper = y_limits
                ax.set_ylim(bottom=lower, top=upper)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            if title and len(method_groups) == 1:
                ax.set_title(title)
            elif title:
                ax.set_title(_panel_title(title, group))
            if plotted:
                ax.legend(loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{ylabel} unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        axes[-1].set_xticklabels(scenarios)
        axes[-1].set_xlabel("Scenario")
        fig.tight_layout()

        filename = f"{filename_base}{suffix}.png"
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    return _render_snir_variants(
        render,
        on_title=f"{ylabel} – SNIR activé",
        off_title=f"{ylabel} – SNIR désactivé",
        mixed_title=f"{ylabel} – SNIR superposé",
    )


def plot_der(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    """Trace la DER globale par scénario en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="der_global",
        ylabel="Global DER",
        filename_base="der_global_vs_scenarios",
        ci_resolver=_ratio_ci_for_metric,
        per_algo=per_algo,
    )


def plot_pdr(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    """Trace le PDR global en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="pdr_global",
        ylabel="Global PDR",
        filename_base="pdr_global_vs_scenarios",
        ci_resolver=_ratio_ci_for_metric,
        per_algo=per_algo,
    )


def plot_snir_mean(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    """Trace le SNIR moyen en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="snir_mean",
        ylabel="SNIR moyen (dB)",
        filename_base="snir_mean_vs_scenarios",
        y_limits=None,
        ci_resolver=_snir_ci_for_metric,
        per_algo=per_algo,
    )


def plot_snir_moving_average(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    window_size: int,
) -> List[Path]:
    """Trace la moyenne mobile du SNIR moyen en distinguant l'état SNIR."""

    if not scenarios or not metrics_by_method:
        return []

    window_size = max(1, int(window_size))
    x_positions = list(range(len(scenarios)))
    values_by_state = _values_by_snir_state(metrics_by_method, scenarios, "snir_mean")

    aggregated_by_state: Dict[str, List[float]] = {}
    for state, method_values in values_by_state.items():
        scenario_values: List[float] = []
        for idx in range(len(scenarios)):
            values = [
                series[idx]
                for series in method_values.values()
                if idx < len(series) and not math.isnan(series[idx])
            ]
            scenario_values.append(float(np.mean(values)) if values else float("nan"))
        aggregated_by_state[state] = _moving_average(scenario_values, window_size)

    window_label = f"fenêtre {window_size} scénarios"

    def render(states_to_plot: List[str], suffix: str, title: str) -> Optional[Path]:
        fig, ax = plt.subplots(figsize=(max(6.0, 2.5 * len(scenarios)), 4.5))
        plotted = False
        for state in states_to_plot:
            values = aggregated_by_state.get(state, [])
            if not values or _all_nan(values):
                continue
            color = SNIR_STATE_COLORS.get(state, "#7f7f7f")
            label_state = SNIR_STATE_LABELS.get(state, state)
            marker = "o" if state == "snir_on" else "s"
            ax.plot(
                x_positions,
                values,
                marker=marker,
                color=color,
                label=label_state,
            )
            plotted = True

        ax.set_xticks(x_positions, scenarios)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("SNIR moyen (dB)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_title(title or f"SNIR moyen mobile ({window_label})")
        if plotted:
            ax.legend(loc="best", title=window_label)
        else:
            ax.text(
                0.5,
                0.5,
                "SNIR moyen indisponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()

        filename = f"snir_moving_average_vs_scenarios{suffix}.png"
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    return _render_snir_variants(
        render,
        on_title=f"SNIR moyen mobile – SNIR activé ({window_label})",
        off_title=f"SNIR moyen mobile – SNIR désactivé ({window_label})",
        mixed_title=f"SNIR moyen mobile – superposé ({window_label})",
    )


def plot_energy_snir(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    """Trace l'énergie totale par scénario en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="energy_j",
        ylabel="Énergie cumulée (J)",
        filename_base="energy_total_vs_scenarios",
        y_limits=(0.0, None),
        per_algo=per_algo,
    )


def plot_jain_index_snir(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    """Trace l'indice de Jain par scénario en distinguant l'état SNIR."""

    return _plot_metric_with_snir_states(
        metrics_by_method,
        scenarios,
        out_dir,
        attribute="jain_index",
        ylabel="Indice de Jain",
        filename_base="jain_index_vs_scenarios",
        per_algo=per_algo,
    )


def plot_collisions(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    x_positions = list(range(len(scenarios)))
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    collision_values = _values_for_attribute(metrics_by_method, scenarios, "collisions")
    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = collision_values.get(method, [])
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_positions, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_xticks(x_positions)
        ax.set_ylabel("Uplink collisions")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0.0)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "Collision data unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xticklabels(scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "collisions_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_energy(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    x_positions = list(range(len(scenarios)))
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    energy_values = _values_for_attribute(metrics_by_method, scenarios, "energy_j")
    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = energy_values.get(method, [])
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_positions, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_xticks(x_positions)
        ax.set_ylabel("Total energy (J)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0.0)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "Energy data unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xticklabels(scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "energy_total_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_jain_index(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    x_positions = list(range(len(scenarios)))
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    jain_values = _values_for_attribute(metrics_by_method, scenarios, "jain_index")
    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = jain_values.get(method, [])
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_positions, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_xticks(x_positions)
        ax.set_ylabel("Jain index")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "Jain index unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xticklabels(scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "jain_index_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_min_sf_share(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    x_positions = list(range(len(scenarios)))
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(max(6.0, 2.5 * len(scenarios)), 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    min_sf_values = _values_for_attribute(metrics_by_method, scenarios, "min_sf_share")
    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            values = min_sf_values.get(method, [])
            if not values or _all_nan(values):
                continue
            color, marker = method_styles[method]
            ax.plot(x_positions, values, marker=marker, color=color, label=method)
            plotted = True

        ax.set_xticks(x_positions)
        ax.set_ylabel("Minimum SF share")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "SF distribution unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xticklabels(scenarios)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()

    output_path = out_dir / "min_sf_share_vs_scenarios.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def sanitize_filename(text: str) -> str:
    """Transforme un scénario en identifiant de fichier valide."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return cleaned.strip("_") or "scenario"


def plot_snir_cdf(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    """Génère les CDF SNIR/SNR par scénario avec une courbe par méthode."""

    saved_paths: List[Path] = []
    method_styles = _style_mapping(sorted(metrics_by_method.keys()))

    def _plot_cdf(
        *,
        metric_label: str,
        filename_prefix: str,
        cdf_getter: Callable[[MethodScenarioMetrics], List[Tuple[float, float]]],
        label_suffix: Optional[str] = None,
    ) -> None:
        for scenario in scenarios:
            def render(state_filter: List[str], suffix: str, title: str) -> Optional[Path]:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                has_data = False
                for method in sorted(metrics_by_method.keys()):
                    metric = metrics_by_method[method].get(scenario)
                    if not metric:
                        continue
                    cdf = cdf_getter(metric)
                    if not cdf:
                        continue
                    state = _metric_snir_state(metric)
                    if state_filter and state not in state_filter:
                        continue
                    has_data = True
                    xs, ys = zip(*sorted(cdf))
                    _, marker = method_styles[method]
                    color = SNIR_STATE_COLORS.get(state, "#7f7f7f")
                    label_state = SNIR_STATE_LABELS.get(state, state)
                    label = f"{method} ({label_state})"
                    if label_suffix:
                        label = f"{method} ({label_state}, {label_suffix})"
                    ax.step(
                        xs,
                        ys,
                        where="post",
                        label=label,
                        color=color,
                    )
                    ax.plot([], [], marker=marker, color=color, linestyle="", label="")
                if not has_data:
                    ax.text(
                        0.5,
                        0.5,
                        f"{metric_label} data unavailable",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                ax.set_xlabel(f"{metric_label} (dB)")
                ax.set_ylabel("CDF")
                ax.set_ylim(0.0, 1.0)
                ax.set_xlim(auto=True)
                ax.grid(True, linestyle="--", alpha=0.4)
                if title:
                    ax.set_title(title)
                handles, labels = ax.get_legend_handles_labels()
                filtered_handles = [h for h, l in zip(handles, labels) if l]
                filtered_labels = [l for l in labels if l]
                if filtered_handles:
                    ax.legend(filtered_handles, filtered_labels, loc="best")
                fig.tight_layout()

                filename = f"{filename_prefix}_{sanitize_filename(scenario)}{suffix}.png"
                output_path = out_dir / filename
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
                saved_paths.append(output_path)
                return output_path

            _render_snir_variants(
                render,
                on_title=f"{metric_label} CDF – {scenario} (SNIR activé)",
                off_title=f"{metric_label} CDF – {scenario} (SNIR désactivé)",
                mixed_title=f"{metric_label} CDF – {scenario} (superposé)",
            )

    _plot_cdf(
        metric_label="SNIR",
        filename_prefix="snir_cdf",
        cdf_getter=lambda metric: metric.snir_cdf,
    )
    _plot_cdf(
        metric_label="SNR",
        filename_prefix="snr_cdf",
        cdf_getter=lambda metric: metric.snr_cdf,
        label_suffix="SNR",
    )
    return saved_paths


def _compute_pdr_by_snir_bins(
    df: Optional[pd.DataFrame],
    *,
    bin_width: float = 1.0,
    cluster_value: Optional[str] = None,
) -> List[Tuple[float, float]]:
    if df is None or df.empty:
        return []
    filtered = df
    if cluster_value is not None:
        cluster_column = _find_column(
            list(df.columns),
            ["cluster", "cluster_id", "clusterId", "ring", "qos_cluster"],
        )
        if cluster_column is None:
            return []
        filtered = df[df[cluster_column].astype(str) == str(cluster_value)]
        if filtered.empty:
            return []
    snir_series = _extract_snir_series(filtered)
    success_series = _extract_success_series(filtered)
    if snir_series is None or success_series is None:
        return []
    series = pd.DataFrame({"snir": snir_series, "success": success_series}).dropna()
    if series.empty:
        return []
    values = series["snir"].to_list()
    minimum = math.floor(min(values))
    maximum = math.ceil(max(values))
    if minimum == maximum:
        pdr_value = float(series["success"].mean())
        return [(float(minimum), pdr_value)]
    bin_edges = [minimum + i * bin_width for i in range(int((maximum - minimum) / bin_width) + 1)]
    bin_edges.append(maximum)
    totals = [0 for _ in range(len(bin_edges) - 1)]
    successes = [0.0 for _ in range(len(bin_edges) - 1)]
    for snir_value, success_value in series[["snir", "success"]].itertuples(index=False):
        index = min(int((snir_value - minimum) / bin_width), len(totals) - 1)
        totals[index] += 1
        successes[index] += float(success_value)
    result: List[Tuple[float, float]] = []
    for idx, total in enumerate(totals):
        if total <= 0:
            continue
        center = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0
        result.append((float(center), float(successes[idx] / total)))
    return result


def plot_pdr_vs_snir_by_method(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    saved: List[Path] = []
    if not scenarios or not metrics_by_method:
        return saved

    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)

    for scenario in scenarios:
        fig, axes = plt.subplots(
            nrows=len(method_groups),
            ncols=1,
            sharex=True,
            figsize=(6.5, 4.5 * len(method_groups)),
        )
        if len(method_groups) == 1:
            axes = [axes]  # type: ignore[assignment]
        for ax, group in zip(axes, method_groups):
            plotted = False
            for method in group:
                metric = metrics_by_method[method].get(scenario)
                if metric is None or not metric.pdr_by_snir_bin:
                    continue
                xs, ys = zip(*sorted(metric.pdr_by_snir_bin))
                color, marker = method_styles[method]
                ax.plot(xs, ys, label=method, color=color, marker=marker)
                plotted = True
            ax.set_ylabel("PDR")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(method_groups) > 1:
                ax.set_title(_panel_title(f"PDR vs SNIR – {scenario}", group))
            else:
                ax.set_title(f"PDR vs SNIR – {scenario}")
            if plotted:
                ax.legend(loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "PDR vs SNIR indisponible",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        axes[-1].set_xlabel("SNIR (dB)")
        fig.tight_layout()

        filename = f"pdr_vs_snir_{sanitize_filename(scenario)}.png"
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved.append(output_path)
    return saved


def plot_pdr_vs_snir_by_cluster(
    metrics_root: Path,
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    saved: List[Path] = []
    if not scenarios or not metrics_by_method:
        return saved

    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)

    for scenario in scenarios:
        data_per_method: Dict[str, pd.DataFrame] = {}
        clusters: List[str] = []
        for method in sorted(metrics_by_method.keys()):
            df = _load_packets_df(metrics_root, method, scenario)
            if df is None or df.empty:
                continue
            data_per_method[method] = df
            cluster_column = _find_column(
                list(df.columns),
                ["cluster", "cluster_id", "clusterId", "ring", "qos_cluster"],
            )
            if cluster_column is None:
                continue
            cluster_values = (
                df[cluster_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            clusters.extend(cluster_values)

        unique_clusters = sorted({cluster for cluster in clusters})
        for cluster in unique_clusters:
            fig, axes = plt.subplots(
                nrows=len(method_groups),
                ncols=1,
                sharex=True,
                figsize=(6.5, 4.5 * len(method_groups)),
            )
            if len(method_groups) == 1:
                axes = [axes]  # type: ignore[assignment]
            for ax, group in zip(axes, method_groups):
                plotted = False
                for method in group:
                    df = data_per_method.get(method)
                    if df is None:
                        continue
                    bins = _compute_pdr_by_snir_bins(df, cluster_value=cluster)
                    if not bins:
                        continue
                    xs, ys = zip(*sorted(bins))
                    color, marker = method_styles[method]
                    ax.plot(xs, ys, label=method, color=color, marker=marker)
                    plotted = True
                ax.set_ylabel("PDR")
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, linestyle="--", alpha=0.4)
                title = f"PDR vs SNIR – {scenario} (cluster {cluster})"
                if len(method_groups) > 1:
                    ax.set_title(_panel_title(title, group))
                else:
                    ax.set_title(title)
                if plotted:
                    ax.legend(loc="best")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "PDR vs SNIR indisponible",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            axes[-1].set_xlabel("SNIR (dB)")
            fig.tight_layout()

            filename = f"pdr_vs_snir_{sanitize_filename(scenario)}_cluster_{sanitize_filename(cluster)}.png"
            output_path = out_dir / filename
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            saved.append(output_path)
    return saved


def plot_collision_rates_vs_nodes(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None
    ordered = _ordered_scenarios_by_nodes(scenarios, node_counts)
    if not ordered:
        return None

    x_values = [node_counts[scenario] for scenario in ordered]
    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(7.0, 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            destructive_values = []
            captured_values = []
            for scenario in ordered:
                metric = metrics_by_method.get(method, {}).get(scenario)
                if metric is None or metric.attempted <= 0:
                    destructive_values.append(float("nan"))
                    captured_values.append(float("nan"))
                    continue
                destructive = metric.collision_destructive
                captured = metric.collision_captured
                destructive_values.append(
                    float(destructive) / metric.attempted if destructive is not None else float("nan")
                )
                captured_values.append(
                    float(captured) / metric.attempted if captured is not None else float("nan")
                )
            color, marker = method_styles[method]
            if not _all_nan(destructive_values):
                ax.plot(
                    x_values,
                    destructive_values,
                    marker=marker,
                    color=color,
                    linestyle="-",
                    label=f"{method} (destructive)",
                )
                plotted = True
            if not _all_nan(captured_values):
                ax.plot(
                    x_values,
                    captured_values,
                    marker=marker,
                    color=color,
                    linestyle="--",
                    label=f"{method} (capture)",
                )
                plotted = True

        ax.set_ylabel("Taux de collisions")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if len(method_groups) > 1:
            ax.set_title(_panel_title("", group))
        if plotted:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.text(
                0.5,
                0.5,
                "Taux de collisions indisponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xlabel("Nombre de nœuds")
    fig.tight_layout()

    output_path = out_dir / "collision_rates_vs_nodes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_energy_vs_qos(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    node_counts: Mapping[str, Optional[int]],
    out_dir: Path,
    *,
    per_algo: bool = False,
) -> List[Path]:
    saved: List[Path] = []
    if not scenarios or not metrics_by_method:
        return saved

    methods = sorted(metrics_by_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)

    def _plot(y_attr: str, ylabel: str, filename: str) -> Optional[Path]:
        fig, axes = plt.subplots(
            nrows=len(method_groups),
            ncols=1,
            sharex=True,
            figsize=(6.5, 4.5 * len(method_groups)),
        )
        if len(method_groups) == 1:
            axes = [axes]  # type: ignore[assignment]
        for ax, group in zip(axes, method_groups):
            plotted = False
            for method in group:
                xs = []
                ys = []
                for scenario in scenarios:
                    metric = metrics_by_method.get(method, {}).get(scenario)
                    node_count = node_counts.get(scenario)
                    if metric is None or node_count is None or node_count <= 0:
                        continue
                    if metric.energy_j is None:
                        continue
                    energy_per_node = metric.energy_j / node_count
                    y_value = getattr(metric, y_attr, None)
                    if y_value is None:
                        continue
                    xs.append(float(energy_per_node))
                    ys.append(float(y_value))
                if not xs or not ys:
                    continue
                color, marker = method_styles[method]
                ax.plot(xs, ys, marker=marker, color=color, linestyle="", label=method)
                plotted = True
            ax.set_ylabel(ylabel)
            if y_attr in {"pdr_global"}:
                ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(method_groups) > 1:
                ax.set_title(_panel_title("", group))
            if plotted:
                ax.legend(loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Données énergie/QoS indisponibles",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        axes[-1].set_xlabel("Énergie moyenne par nœud (J)")
        fig.tight_layout()
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    output = _plot("pdr_global", "PDR global", "energy_per_node_vs_pdr.png")
    if output:
        saved.append(output)
    output = _plot("snir_mean", "SNIR moyen (dB)", "energy_per_node_vs_snir.png")
    if output:
        saved.append(output)
    return saved


def plot_effective_sf_vs_distance(
    metrics_root: Path,
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    scenarios: Sequence[str],
    out_dir: Path,
) -> Optional[Path]:
    if not scenarios or not metrics_by_method:
        return None

    distances: Dict[str, float] = {}
    for scenario in scenarios:
        distance_value = None
        for method in sorted(metrics_by_method.keys()):
            metric = metrics_by_method.get(method, {}).get(scenario)
            if metric and metric.distance_to_gw is not None:
                distance_value = float(metric.distance_to_gw)
                break
        if distance_value is not None:
            distances[scenario] = distance_value

    if not distances:
        return None

    category_values: Dict[str, Dict[str, List[float]]] = {
        "ADR": {},
        "SNIR-aware": {},
    }

    for method in sorted(metrics_by_method.keys()):
        for scenario in scenarios:
            metric = metrics_by_method.get(method, {}).get(scenario)
            if metric is None:
                continue
            nodes_df = _load_nodes_df(metrics_root, method, scenario)
            mean_sf = _extract_mean_sf(nodes_df)
            if mean_sf is None:
                continue
            category: Optional[str] = None
            if "adr" in method.lower():
                category = "ADR"
            elif _metric_snir_state(metric) == "snir_on":
                category = "SNIR-aware"
            if category is None:
                continue
            category_values.setdefault(category, {})
            category_values[category].setdefault(scenario, []).append(mean_sf)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = False
    category_styles = {
        "ADR": (COLORS[0], "o"),
        "SNIR-aware": (COLORS[3], "s"),
    }

    for category, scenario_values in category_values.items():
        points: List[Tuple[float, float]] = []
        for scenario, values in scenario_values.items():
            if scenario not in distances or not values:
                continue
            points.append((distances[scenario], float(np.mean(values))))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys = zip(*points)
        color, marker = category_styles.get(category, (COLORS[0], "o"))
        ax.plot(xs, ys, marker=marker, color=color, label=category)
        plotted = True

    ax.set_xlabel("Distance moyenne au GW (m)")
    ax.set_ylabel("SF effectif moyen")
    ax.grid(True, linestyle="--", alpha=0.4)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "SF effectif indisponible",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()

    output_path = out_dir / "effective_sf_vs_distance.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_rolling_metric_for_scenario(
    metric: str,
    data_per_method: Mapping[str, pd.DataFrame],
    scenario: str,
    out_dir: Path,
    *,
    window_size: float,
    window_mode: str,
    per_algo: bool = False,
) -> Optional[Path]:
    if not data_per_method:
        return None

    methods = sorted(data_per_method.keys())
    method_styles = _style_mapping(methods)
    method_groups = _method_groups(methods, per_algo)
    fig, axes = plt.subplots(
        nrows=len(method_groups),
        ncols=1,
        sharex=True,
        figsize=(7.0, 4.5 * len(method_groups)),
    )
    if len(method_groups) == 1:
        axes = [axes]  # type: ignore[assignment]

    label_map = {
        "pdr": "PDR",
        "der": "DER",
        "snir": "SNIR (dB)",
    }

    x_label = "Temps (s)" if window_mode == "duration" else "Position des paquets émis"
    window_label = f"fenêtre {window_size:g} {'s' if window_mode == 'duration' else 'paquets'}"
    title_metric = label_map.get(metric, metric)

    for ax, group in zip(axes, method_groups):
        plotted = False
        for method in group:
            df = data_per_method.get(method)
            if df is None or df.empty or metric not in df.columns:
                continue
            color, marker = method_styles[method]
            lower = df.get(f"{metric}_low")
            upper = df.get(f"{metric}_high")
            ax.plot(
                df["x"],
                df[metric],
                label=method,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4,
            )
            if lower is not None and upper is not None:
                ax.fill_between(df["x"], lower, upper, color=color, alpha=0.15)
            plotted = True

        ax.set_ylabel(label_map.get(metric, metric))
        if metric in {"pdr", "der"}:
            ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        base_title = f"{title_metric} moyenne glissante – {scenario} ({window_label})"
        if len(method_groups) > 1:
            ax.set_title(_panel_title(base_title, group))
        else:
            ax.set_title(base_title)
        if plotted:
            legend = ax.legend(loc="best", title=window_label)
            if legend is not None:
                legend.set_title(window_label)
                legend.get_title().set_text(window_label)
                legend.get_title().set_fontsize("small")
        else:
            ax.text(
                0.5,
                0.5,
                "Données temporelles indisponibles",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    filename = f"rolling_{metric}_{sanitize_filename(scenario)}_{window_mode}.png"
    output_path = out_dir / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_rolling_qos(
    metrics_by_method: Mapping[str, Mapping[str, MethodScenarioMetrics]],
    metrics_root: Path,
    scenarios: Sequence[str],
    out_dir: Path,
    *,
    window_size: float,
    window_mode: str,
    per_algo: bool = False,
) -> List[Path]:
    saved: List[Path] = []
    methods = sorted(metrics_by_method.keys())
    require_time = window_mode == "duration"

    for scenario in scenarios:
        data_per_method: Dict[str, pd.DataFrame] = {}
        for method in methods:
            timeseries = _load_packet_timeseries(
                metrics_root, method, scenario, require_time=require_time
            )
            if timeseries is None or timeseries.empty:
                continue
            rolling_df = _rolling_metrics(timeseries, window_size, window_mode)
            data_per_method[method] = rolling_df

        for metric in ["pdr", "der", "snir"]:
            output = _plot_rolling_metric_for_scenario(
                metric,
                data_per_method,
                scenario,
                out_dir,
                window_size=window_size,
                window_mode=window_mode,
                per_algo=per_algo,
            )
            if output:
                saved.append(output)

    return saved


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics_root = args.root
    config_path: Optional[Path] = args.config
    out_dir: Path = args.out

    if not metrics_root.exists():
        raise FileNotFoundError(f"Dossier de résultats introuvable : {metrics_root}")

    scenarios_cfg: Optional[Mapping[str, Mapping[str, object]]] = None
    gateway_position: Optional[Tuple[float, float]] = None
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")
        scenarios_cfg = load_yaml_config(config_path)
        gateway_position = load_gateway_position(config_path)

    all_metrics = load_all_metrics(metrics_root, scenarios_cfg, gateway_position=gateway_position)
    if not all_metrics:
        raise RuntimeError("Aucune métrique détectée – vérifiez la structure des résultats.")

    metrics_by_method = build_method_mapping(all_metrics)
    scenarios = ordered_scenarios(all_metrics, scenarios_cfg)
    if not scenarios:
        scenarios = sorted({scenario for _, scenario in all_metrics.keys()})
    node_counts = _scenario_node_counts(metrics_by_method, scenarios, metrics_root, scenarios_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_pdr(metrics_by_method, scenarios, out_dir)
    plot_pdr(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_der(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_snir_mean(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_snir_moving_average(
        metrics_by_method,
        scenarios,
        out_dir,
        window_size=int(args.moving_average_window),
    )
    plot_collisions(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_energy(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_energy_snir(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_jain_index(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_jain_index_snir(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_min_sf_share(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_snir_cdf(metrics_by_method, scenarios, out_dir)
    plot_pdr_vs_nodes(metrics_by_method, scenarios, node_counts, out_dir, per_algo=args.per_algo)
    plot_der_vs_nodes(metrics_by_method, scenarios, node_counts, out_dir, per_algo=args.per_algo)
    plot_cluster_pdr_vs_nodes(metrics_by_method, scenarios, node_counts, out_dir)
    plot_snir_mean_vs_nodes_by_sf(metrics_by_method, scenarios, node_counts, out_dir)
    plot_snir_mean_vs_nodes_by_cluster(metrics_by_method, scenarios, node_counts, out_dir)
    plot_pdr_vs_snir_by_method(metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_pdr_vs_snir_by_cluster(metrics_root, metrics_by_method, scenarios, out_dir, per_algo=args.per_algo)
    plot_effective_sf_vs_distance(metrics_root, metrics_by_method, scenarios, out_dir)
    plot_collision_rates_vs_nodes(metrics_by_method, scenarios, node_counts, out_dir, per_algo=args.per_algo)
    plot_energy_vs_qos(metrics_by_method, scenarios, node_counts, out_dir, per_algo=args.per_algo)
    plot_rolling_qos(
        metrics_by_method,
        metrics_root,
        scenarios,
        out_dir,
        window_size=float(args.rolling_window),
        window_mode=str(args.window_mode),
        per_algo=args.per_algo,
    )


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    main()
