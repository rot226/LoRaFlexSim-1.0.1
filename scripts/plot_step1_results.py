"""Génère les figures de l'étape 1 à partir des CSV produits par Tâche 4.

Les figures standard (dans figures/step1/) ne sont plus recommandées ; privilégiez
les figures étendues dans figures/step1/extended/. Pour les comparaisons
multi-algorithmes 3×2 / 1×2 décrites dans README_FIGURES.md, utilisez plutôt
scripts/plot_step1_comparison.py et gardez --compare-snir activé.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - dépend de l'environnement de test
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.ticker import MaxNLocator, ScalarFormatter  # type: ignore
except Exception:  # pragma: no cover - permet de continuer même sans matplotlib
    plt = None  # type: ignore

from plot_theme import (
    SNIR_COLORS,
    THEME_LABEL_SIZE,
    THEME_LINE_WIDTH,
    THEME_MARKER_EDGE_WIDTH,
    THEME_MARKER_SIZE,
    THEME_TICK_LABEL_SIZE,
    THEME_TITLE_SIZE,
    apply_plot_theme,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures"

__all__ = [
    "generate_step1_figures",
    "plot_distribution_by_state",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_FIGURES_DIR",
]

STATE_LABELS = {True: "snir_on", False: "snir_off", None: "snir_unknown"}
SNIR_LABELS = {
    "snir_on": "SNIR enabled",
    "snir_off": "SNIR disabled",
    "snir_unknown": "SNIR unknown",
}
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]
MIXRA_OPT_ALIASES = {"mixra_opt", "mixraopt", "mixra-opt", "mixra opt", "opt"}
MIXRA_H_ALIASES = {"mixra_h", "mixrah", "mixra-h", "mixra h"}
EXTENDED_ALGORITHM = "mixra_opt"


def _normalize_algorithm_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if normalized in MIXRA_OPT_ALIASES:
        return "mixra_opt"
    if normalized in MIXRA_H_ALIASES:
        return "mixra_h"
    return text


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


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


def _detect_snir_state(row: Mapping[str, Any]) -> Tuple[str | None, bool]:
    snir_state_raw = row.get("snir_state")
    if snir_state_raw is not None and str(snir_state_raw).strip() != "":
        normalized = str(snir_state_raw).strip().lower()
        if normalized in {"snir_on", "on", "true", "1", "yes", "y"}:
            return "snir_on", True
        if normalized in {"snir_off", "off", "false", "0", "no", "n"}:
            return "snir_off", True
        if normalized in {"snir_unknown", "unknown", "na", "n/a"}:
            return "snir_unknown", True
        return None, False

    parsed = _parse_bool(row.get("use_snir"))
    if parsed is not None:
        return STATE_LABELS.get(parsed, "snir_unknown"), True
    return None, False


def _record_matches_state(record: Mapping[str, Any], state: str) -> bool:
    return record.get("snir_state") == state and record.get("snir_detected", True)


def _select_signal_mean(record: Mapping[str, Any]) -> Tuple[float | None, str]:
    use_snir = record.get("use_snir")
    if use_snir is True:
        return _maybe_float(record.get("snir_mean")), "snir_mean"
    if use_snir is False:
        return _maybe_float(record.get("snr_mean")), "snr_mean"
    snir_value = _maybe_float(record.get("snir_mean"))
    if snir_value is not None:
        return snir_value, "snir_mean"
    return _maybe_float(record.get("snr_mean")), "snr_mean"


def _load_step1_records(results_dir: Path, strict: bool = False) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    snir_unknown_rows = 0
    if not results_dir.exists():
        return records
    for csv_path in sorted(results_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            if strict:
                required_columns = {"snir_state", "snir_mean", "snir_histogram_json"}
                fieldnames = set(reader.fieldnames or [])
                if not required_columns.issubset(fieldnames):
                    warnings.warn(
                        (
                            "CSV ignoré (filtrage strict) : "
                            f"{csv_path} ne contient pas {sorted(required_columns)}."
                        ),
                        RuntimeWarning,
                    )
                    continue
            for row in reader:
                cluster_pdr: Dict[int, float] = {}
                cluster_targets: Dict[int, float] = {}
                cluster_der: Dict[int, float] = {}
                for key, value in row.items():
                    if key.startswith("qos_cluster_pdr__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_pdr[cluster_id] = _parse_float(value)
                    elif key.startswith("qos_cluster_targets__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_targets[cluster_id] = _parse_float(value)
                    elif key.startswith("qos_cluster_der__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_der[cluster_id] = _parse_float(value)

                snir_candidate = row.get("snir_mean")
                snr_candidate = row.get("snr_mean") or row.get("SNR") or row.get("snr")
                algorithm = _normalize_algorithm_name(row.get("algorithm"))
                if not algorithm:
                    algorithm = _normalize_algorithm_name(csv_path.parent.name) or csv_path.parent.name
                record: Dict[str, Any] = {
                    "csv_path": csv_path,
                    "algorithm": algorithm,
                    "num_nodes": int(float(row.get("num_nodes", "0") or 0)),
                    "packet_interval_s": float(row.get("packet_interval_s", "0") or 0),
                    "random_seed": _maybe_int(row.get("random_seed") or row.get("seed")),
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "snir_mean": _maybe_float(snir_candidate),
                    "snr_mean": _maybe_float(snr_candidate),
                    "collisions": int(float(row.get("collisions", "0") or 0)),
                    "collisions_snir": int(float(row.get("collisions_snir", "0") or 0)),
                    "jain_index": _parse_float(row.get("jain_index")),
                    "throughput_bps": _parse_float(row.get("throughput_bps")),
                    "cluster_pdr": cluster_pdr,
                    "cluster_der": cluster_der,
                    "cluster_targets": cluster_targets,
                }
                snir_state, snir_detected = _detect_snir_state(row)
                if not snir_detected:
                    warnings.warn(
                        f"Aucun état SNIR explicite dans {csv_path}; la ligne sera ignorée pour les figures mixtes.",
                        RuntimeWarning,
                    )
                    continue
                record["use_snir"] = True if snir_state == "snir_on" else False if snir_state == "snir_off" else None
                record["snir_state"] = snir_state
                record["snir_detected"] = snir_detected
                if snir_state == "snir_unknown":
                    snir_unknown_rows += 1
                if record["use_snir"] is True and record.get("snir_mean") is None:
                    raise ValueError(
                        f"SNIR activé sans snir_mean dans {csv_path} (seed {record.get('random_seed')})."
                    )
                records.append(record)
    if snir_unknown_rows:
        warnings.warn(
            (
                f"{snir_unknown_rows} ligne(s) snir_unknown détectée(s) dans {results_dir}; "
                "elles sont exclues des figures mixtes."
            ),
            RuntimeWarning,
        )
    return records


def _snir_label(state: str | None) -> str:
    return SNIR_LABELS.get(state or "snir_unknown", SNIR_LABELS["snir_unknown"])


def _snir_color(state: str | None) -> str:
    return SNIR_COLORS.get(state or "snir_unknown", SNIR_COLORS["snir_unknown"])


def _unique_algorithms(records: Iterable[Mapping[str, Any]]) -> List[str]:
    algorithms: set[str] = set()
    for record in records:
        raw_value = record.get("algorithm")
        normalized = _normalize_algorithm_name(raw_value)
        if normalized:
            algorithms.add(normalized)
        elif raw_value:
            algorithms.add(str(raw_value))
        else:
            algorithms.add("unknown")
    return sorted(algorithms)


def _filter_records_for_algorithm(
    records: Iterable[Mapping[str, Any]], algorithm: str | None
) -> List[Dict[str, Any]]:
    if not algorithm:
        return [dict(record) for record in records]
    normalized = _normalize_algorithm_name(algorithm) or str(algorithm)
    filtered = [
        dict(record)
        for record in records
        if _normalize_algorithm_name(record.get("algorithm")) == normalized
    ]
    if not filtered:
        warnings.warn(
            f"Aucune donnée trouvée pour l'algorithme {normalized}.",
            RuntimeWarning,
        )
    return filtered


def _render_snir_variants(
    render: Any,
    *,
    on_title: str,
    off_title: str,
    mixed_title: str,
) -> None:
    variants = [
        (["snir_on"], "_snir-on", on_title),
        (["snir_off"], "_snir-off", off_title),
        (["snir_on", "snir_off"], "_snir-mixed", mixed_title),
    ]
    for states, suffix, title in variants:
        filtered_states = states
        if suffix == "_snir-mixed":
            filtered_states = [state for state in states if state != "snir_unknown"]
        render(filtered_states, suffix, title)


def _format_axes(ax: Any, integer_x: bool = False) -> None:
    if plt is None:
        return
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.5)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(
        axis="both",
        direction="in",
        length=4.5,
        width=1.0,
        labelsize=THEME_TICK_LABEL_SIZE,
    )
    ax.xaxis.label.set_size(THEME_LABEL_SIZE)
    ax.yaxis.label.set_size(THEME_LABEL_SIZE)
    ax.title.set_size(THEME_TITLE_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for line in ax.get_lines():
        line.set_linewidth(THEME_LINE_WIDTH)
        line.set_markersize(THEME_MARKER_SIZE)
        line.set_markeredgewidth(THEME_MARKER_EDGE_WIDTH)


def _metric_error_bounds(record: Mapping[str, Any], metric: str, value: float) -> Tuple[float, float] | None:
    for base in {metric, metric.lower(), metric.upper()}:
        for prefix in (f"{base}_ci_low", f"{base}_ci95_low", f"{base}_ci_95_low"):
            ci_low = _maybe_float(record.get(prefix))
            if ci_low is not None:
                break
        else:
            ci_low = None
        for prefix in (f"{base}_ci_high", f"{base}_ci95_high", f"{base}_ci_95_high"):
            ci_high = _maybe_float(record.get(prefix))
            if ci_high is not None:
                break
        else:
            ci_high = None
        if ci_low is not None and ci_high is not None:
            return max(0.0, value - ci_low), max(0.0, ci_high - value)
        std = _maybe_float(record.get(f"{base}_std"))
        if std is not None:
            return float(std), float(std)
    return None


def _load_summary_records(summary_path: Path, forced_state: str | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    snir_unknown_rows = 0
    if not summary_path.exists():
        return records

    with summary_path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record: Dict[str, Any] = {"summary_path": summary_path}
            for key, value in row.items():
                if key in {"algorithm", "snir_state"}:
                    record[key] = value
                elif key in {"random_seed"}:
                    record[key] = _maybe_int(value)
                elif key in {"num_nodes"}:
                    record[key] = int(float(value or 0))
                elif key in {"packet_interval_s"}:
                    record[key] = float(value or 0)
                else:
                    record[key] = _parse_float(value)
            record["algorithm"] = _normalize_algorithm_name(record.get("algorithm")) or record.get("algorithm")
            snir_state, snir_detected = _detect_snir_state(row)
            if snir_detected and not record.get("snir_state"):
                record["snir_state"] = snir_state
            if forced_state and not record.get("snir_state"):
                record["snir_state"] = forced_state
                snir_detected = True
            if not snir_detected:
                warnings.warn(
                    f"Aucun état SNIR explicite dans {summary_path}; la ligne sera ignorée pour les figures mixtes.",
                    RuntimeWarning,
                )
                continue
            if record.get("snir_state") == "snir_on":
                record["use_snir"] = True
            elif record.get("snir_state") == "snir_off":
                record["use_snir"] = False
            record["snir_detected"] = snir_detected
            if record.get("snir_state") == "snir_unknown":
                snir_unknown_rows += 1
            records.append(record)
    if snir_unknown_rows:
        warnings.warn(
            (
                f"{snir_unknown_rows} ligne(s) snir_unknown détectée(s) dans {summary_path}; "
                "elles sont exclues des figures mixtes."
            ),
            RuntimeWarning,
        )
    return records


def _load_comparison_records(results_dir: Path, use_summary: bool, strict: bool) -> List[Dict[str, Any]]:
    if use_summary:
        explicit_on = _load_summary_records(results_dir / "summary_snir_on.csv", forced_state="snir_on")
        explicit_off = _load_summary_records(results_dir / "summary_snir_off.csv", forced_state="snir_off")
        combined = _load_summary_records(results_dir / "summary.csv")
        records = explicit_on + explicit_off + combined
    else:
        records = _load_step1_records(results_dir, strict=strict)
    return records


def _load_raw_samples(raw_path: Path, fallback_dir: Path, strict: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if raw_path.exists():
        with raw_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                algorithm = _normalize_algorithm_name(row.get("algorithm")) or row.get("algorithm")
                record: Dict[str, Any] = {
                    "algorithm": algorithm,
                    "snir_state": row.get("snir_state", "snir_unknown"),
                    "packet_interval_s": _parse_float(row.get("packet_interval_s")),
                    "DER": _parse_float(row.get("DER")),
                }
                records.append(record)
    else:
        records = _load_step1_records(fallback_dir, strict=strict)
    return records


def _apply_ieee_filters(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [
        record
        for record in records
        if record.get("snir_state") in {"snir_on", "snir_off"}
        and record.get("snir_detected", True)
    ]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in filtered:
        key = (
            record.get("algorithm"),
            record.get("num_nodes"),
            record.get("packet_interval_s"),
            record.get("random_seed"),
            record.get("simulation_duration_s"),
        )
        groups[key].append(record)

    coherent_records: List[Dict[str, Any]] = []
    for items in groups.values():
        states = {item.get("snir_state") for item in items}
        if {"snir_on", "snir_off"}.issubset(states):
            coherent_records.extend(items)

    dropped = len(records) - len(coherent_records)
    if dropped > 0:
        warnings.warn(
            f"Filtre IEEE : {dropped} enregistrement(s) incohérents ou SNIR inconnus exclus.",
            RuntimeWarning,
        )
    return coherent_records


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
    algorithms = _unique_algorithms(records)
    error_metrics = {"PDR", "DER", "snir_mean", "snr_mean", "SNIR", "SNR"}

    def render(states: List[str], suffix: str, title: str) -> None:
        for period in periods:
            fig, ax = plt.subplots(figsize=(6, 4))
            for state in states:
                state_records = [r for r in records if _record_matches_state(r, state)]
                if not state_records:
                    continue
                for algo_idx, algorithm in enumerate(algorithms):
                    data = [
                        r
                        for r in state_records
                        if r["algorithm"] == algorithm and r["packet_interval_s"] == period
                    ]
                    data.sort(key=lambda item: item["num_nodes"])
                    xs: List[int] = []
                    ys: List[float] = []
                    used_items: List[Dict[str, Any]] = []
                    used_error_metrics: List[str] = []
                    for item in data:
                        if metric == "snir_mean":
                            raw_value, error_metric = _select_signal_mean(item)
                        else:
                            raw_value = item.get(metric)
                            error_metric = metric
                        if raw_value is None:
                            continue
                        xs.append(item["num_nodes"])
                        ys.append(_parse_float(raw_value))
                        used_items.append(item)
                        used_error_metrics.append(error_metric)
                    if not xs:
                        continue
                    marker = MARKER_CYCLE[algo_idx % len(MARKER_CYCLE)]
                    label = (
                        f"{algorithm} ({_snir_label(state)})"
                        if len(states) > 1
                        else algorithm
                    )
                    if metric in error_metrics:
                        lower: List[float] = []
                        upper: List[float] = []
                        for item, value, error_metric in zip(used_items, ys, used_error_metrics):
                            error = _metric_error_bounds(item, error_metric, value)
                            if error is None:
                                lower.append(float("nan"))
                                upper.append(float("nan"))
                            else:
                                low, high = error
                                lower.append(low)
                                upper.append(high)
                        has_errors = any(not (val != val) and val > 0 for val in lower + upper)
                        if has_errors:
                            ax.errorbar(
                                xs,
                                ys,
                                yerr=[lower, upper],
                                marker=marker,
                                markersize=5.5,
                                linewidth=2,
                                color=_snir_color(state),
                                label=label,
                                capsize=4,
                            )
                        else:
                            ax.plot(
                                xs,
                                ys,
                                marker=marker,
                                markersize=5.5,
                                linewidth=2,
                                color=_snir_color(state),
                                label=label,
                            )
                    else:
                        ax.plot(
                            xs,
                            ys,
                            marker=marker,
                            markersize=5.5,
                            linewidth=2,
                            color=_snir_color(state),
                            label=label,
                        )
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(ylabel)
            title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            ax.set_title(f"{title} – period {title_period} s")
            _format_axes(ax, integer_x=True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"step1_{filename_prefix}{suffix}_tx_{title_period}.png"
            fig.tight_layout()
            fig.savefig(output, dpi=150)
            plt.close(fig)

    _render_snir_variants(
        render,
        on_title=f"{ylabel} – {_snir_label('snir_on')}",
        off_title=f"{ylabel} – {_snir_label('snir_off')}",
        mixed_title=f"{ylabel} – mixed SNIR",
    )


def _plot_summary_bars(
    records: List[Dict[str, Any]],
    figures_dir: Path,
    forced_algorithm: str | None = None,
) -> None:
    if not records or plt is None:
        return

    if forced_algorithm:
        records = _filter_records_for_algorithm(records, forced_algorithm)
        if not records:
            return

    metrics = {
        "PDR": "Overall PDR",
        "DER": "Overall DER",
        "snir_mean": "Mean SNIR (dB)",
        "snr_mean": "Mean SNR (dB)",
        "collisions": "Collisions",
        "collisions_snir": "Collisions (SNIR)",
        "jain_index": "Jain index",
        "throughput_bps": "Aggregate throughput (bps)",
    }

    periods = sorted({r.get("packet_interval_s") for r in records})
    snir_states = [
        state
        for state in ("snir_on", "snir_off", "snir_unknown")
        if state in {r.get("snir_state") for r in records if r.get("snir_detected", True)}
    ]
    if not snir_states:
        return

    for period in periods:
        filtered = [r for r in records if r.get("packet_interval_s") == period]
        if not filtered:
            continue
        combinations = sorted({(r.get("num_nodes"), r.get("algorithm")) for r in filtered})
        if not combinations:
            continue

        for metric, ylabel in metrics.items():
            if not any(f"{metric}_mean" in r or metric in r for r in filtered):
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            positions = list(range(len(combinations)))
            width = 0.2 if len(snir_states) > 0 else 0.4

            for idx, state in enumerate(snir_states):
                offsets = [p + (idx - (len(snir_states) - 1) / 2) * width for p in positions]
                values: List[float] = []
                errors: List[float] = []
                lower_errors: List[float] = []
                upper_errors: List[float] = []
                for combo in combinations:
                    num_nodes, algorithm = combo
                    match = next(
                        (
                            r
                            for r in filtered
                            if r.get("num_nodes") == num_nodes
                            and r.get("algorithm") == algorithm
                            and _record_matches_state(r, state)
                        ),
                        None,
                    )
                    if match:
                        if metric == "snir_mean":
                            signal_value, error_metric = _select_signal_mean(match)
                            values.append(signal_value or 0.0)
                            errors.append(_maybe_float(match.get(f"{error_metric}_std")) or 0.0)
                        elif metric in {"PDR", "DER"}:
                            value = match.get(f"{metric}_mean", 0.0)
                            values.append(value)
                            error = _metric_error_bounds(match, metric, value)
                            if error is None:
                                lower_errors.append(0.0)
                                upper_errors.append(0.0)
                            else:
                                lower, upper = error
                                lower_errors.append(lower)
                                upper_errors.append(upper)
                        else:
                            values.append(match.get(f"{metric}_mean", 0.0))
                            errors.append(_maybe_float(match.get(f"{metric}_std")) or 0.0)
                    else:
                        values.append(0.0)
                        if metric in {"PDR", "DER"}:
                            lower_errors.append(0.0)
                            upper_errors.append(0.0)
                        else:
                            errors.append(0.0)

                if metric in {"PDR", "DER"}:
                    yerr = [lower_errors, upper_errors]
                else:
                    yerr = errors
                ax.bar(
                    offsets,
                    values,
                    width=width,
                    yerr=yerr,
                    label=_snir_label(state),
                    color=_snir_color(state),
                    capsize=4,
                    edgecolor="black",
                    linewidth=0.9,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{algo}\n{nodes} nodes" for nodes, algo in combinations], rotation=0)
            ax.set_ylabel(ylabel)
            period_label = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            ax.set_title(f"{ylabel} – period {period_label} s")
            _format_axes(ax, integer_x=False)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"summary_{metric.lower()}_tx_{period_label}.png"
            fig.tight_layout()
            fig.savefig(output, dpi=150)
            plt.close(fig)


def _plot_cdf(
    records: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    forced_algorithm: str | None = None,
) -> None:
    if not records or plt is None:
        return

    if forced_algorithm:
        records = _filter_records_for_algorithm(records, forced_algorithm)
        if not records:
            return

    by_algorithm: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        algo = str(record.get("algorithm") or "unknown")
        if not record.get("snir_detected", True):
            continue
        state = str(record.get("snir_state") or "snir_unknown")
        der = _parse_float(record.get("DER"))
        by_algorithm[algo][state].append(der)

    for algorithm, state_values in sorted(by_algorithm.items()):
        fig, ax = plt.subplots(figsize=(7, 5))
        for state in ("snir_on", "snir_off", "snir_unknown"):
            values = state_values.get(state, [])
            if not values:
                continue
            sorted_values = sorted(values)
            n = len(sorted_values)
            y = [i / n for i in range(1, n + 1)]
            ax.step(
                sorted_values,
                y,
                where="post",
                label=f"{algorithm} – {_snir_label(state)}",
                color=_snir_color(state),
                linewidth=2,
            )

        ax.set_xlabel("DER")
        ax.set_ylabel("F(x)")
        ax.set_title(f"CDF DER – {algorithm}")
        _format_axes(ax, integer_x=False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = figures_dir / f"cdf_der_{algorithm}.png"
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
    algorithms = _unique_algorithms(records)
    def render(states: List[str], suffix: str, title: str) -> None:
        for period in periods:
            filtered = [
                r
                for r in records
                if r["packet_interval_s"] == period
                and r.get("snir_state") in states
                and r.get("snir_detected", True)
            ]
            if not filtered:
                continue
            fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
            if len(clusters) == 1:
                axes = [axes]
            for idx, cluster_id in enumerate(clusters):
                ax = axes[idx]
                for state in states:
                    state_records = [r for r in filtered if _record_matches_state(r, state)]
                    for algo_idx, algorithm in enumerate(algorithms):
                        algo_records = [r for r in state_records if r["algorithm"] == algorithm]
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
                            marker = MARKER_CYCLE[algo_idx % len(MARKER_CYCLE)]
                            label = (
                                f"{algorithm} ({_snir_label(state)})"
                                if len(states) > 1
                                else algorithm
                            )
                            ax.plot(
                                xs,
                                ys,
                                marker=marker,
                                markersize=5.5,
                                linewidth=2,
                                color=_snir_color(state),
                                label=label,
                            )
                target = None
                for item in filtered:
                    target = item.get("cluster_targets", {}).get(cluster_id)
                    if target is not None:
                        break
                if target is not None:
                    ax.axhline(target, color="black", linestyle="--", linewidth=1, label="Target" if idx == 0 else None)
                ax.set_title(f"Cluster {cluster_id}")
                ax.set_xlabel("Nodes")
                if idx == 0:
                    ax.set_ylabel("PDR")
                ax.set_ylim(0.0, 1.05)
                _format_axes(ax, integer_x=True)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
            title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            fig.suptitle(f"{title} – period {title_period} s")
            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"step1_cluster_pdr{suffix}_tx_{title_period}.png"
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(output, dpi=150)
            plt.close(fig)

    _render_snir_variants(
        render,
        on_title="Cluster PDR – SNIR enabled",
        off_title="Cluster PDR – SNIR disabled",
        mixed_title="Cluster PDR – mixed SNIR",
    )


def _plot_cluster_der(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return
    clusters = sorted({cid for r in records for cid in r.get("cluster_der", {})})
    if not clusters:
        return
    periods = sorted({r["packet_interval_s"] for r in records})
    algorithms = _unique_algorithms(records)
    states = ["snir_on", "snir_off"]

    for period in periods:
        filtered = [
            r
            for r in records
            if r["packet_interval_s"] == period
            and r.get("snir_state") in states
            and r.get("snir_detected", True)
        ]
        if not filtered:
            continue
        fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
        if len(clusters) == 1:
            axes = [axes]
        for idx, cluster_id in enumerate(clusters):
            ax = axes[idx]
            for state in states:
                state_records = [r for r in filtered if _record_matches_state(r, state)]
                for algo_idx, algorithm in enumerate(algorithms):
                    algo_records = [r for r in state_records if r["algorithm"] == algorithm]
                    algo_records.sort(key=lambda item: item["num_nodes"])
                    xs: List[int] = []
                    ys: List[float] = []
                    for item in algo_records:
                        value = item.get("cluster_der", {}).get(cluster_id)
                        if value is None:
                            continue
                        xs.append(item["num_nodes"])
                        ys.append(value)
                    if xs:
                        marker = MARKER_CYCLE[algo_idx % len(MARKER_CYCLE)]
                        label = f"{algorithm} ({_snir_label(state)})"
                        ax.plot(
                            xs,
                            ys,
                            marker=marker,
                            markersize=5.5,
                            linewidth=2,
                            color=_snir_color(state),
                            label=label,
                        )
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Nodes")
            if idx == 0:
                ax.set_ylabel("DER")
            ax.set_ylim(0.0, 1.05)
            _format_axes(ax, integer_x=True)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
        title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
        fig.suptitle(f"DER par cluster – SNIR ON/OFF superposés (period {title_period} s)")
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = figures_dir / f"step1_cluster_der_overlay_tx_{title_period}.png"
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output, dpi=150)
        plt.close(fig)


def _plot_trajectories(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return

    metrics = {"PDR": "PDR", "DER": "DER"}
    def normalize_algorithm(value: Any) -> str:
        normalized = _normalize_algorithm_name(value)
        if normalized:
            return normalized
        if value is not None and str(value).strip():
            return str(value)
        return "unknown"

    algorithms = sorted({normalize_algorithm(r.get("algorithm")) for r in records})
    seeds = sorted({r.get("random_seed") for r in records if r.get("random_seed") is not None})
    if not seeds:
        warnings.warn(
            "Aucune graine détectée pour les trajectoires (colonne random_seed/seed manquante).",
            RuntimeWarning,
        )
        return

    color_map = plt.get_cmap("tab20")
    seed_colors = {
        seed: color_map(idx % max(1, color_map.N)) for idx, seed in enumerate(seeds)
    }
    snir_styles = {
        "snir_on": {"linestyle": "-", "marker": "o"},
        "snir_off": {"linestyle": "--", "marker": "s"},
    }

    def plot_dimension(
        *,
        algorithm: str,
        metric: str,
        ylabel: str,
        x_key: str,
        fixed_key: str,
        fixed_values: Sequence[float],
        x_label: str,
        filename_tag: str,
    ) -> None:
        for fixed_value in fixed_values:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            collected_xs: List[float] = []
            for seed in seeds:
                for state, style in snir_styles.items():
                    subset = [
                        r
                        for r in records
                        if normalize_algorithm(r.get("algorithm")) == algorithm
                        and r.get(fixed_key) == fixed_value
                        and r.get("random_seed") == seed
                        and _record_matches_state(r, state)
                    ]
                    subset.sort(key=lambda item: _parse_float(item.get(x_key)))
                    xs = [_parse_float(item.get(x_key)) for item in subset]
                    ys = [_select_metric_value(item, metric) for item in subset]
                    if not xs:
                        continue
                    collected_xs.extend(xs)
                    label = f"seed {seed} – {_snir_label(state)}"
                    ax.plot(
                        xs,
                        ys,
                        color=seed_colors.get(seed),
                        label=label,
                        linewidth=2,
                        markersize=5.5,
                        **style,
                    )

            if not ax.get_lines():
                plt.close(fig)
                continue

            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            fixed_label = (
                f"{fixed_value:.0f}" if float(fixed_value).is_integer() else f"{fixed_value:g}"
            )
            if fixed_key == "packet_interval_s":
                fixed_desc = f"period {fixed_label} s"
            else:
                fixed_desc = f"{fixed_label} nodes"
            ax.set_title(
                f"Trajectories {ylabel} – {algorithm} – {fixed_desc}"
            )
            integer_x = all(float(x).is_integer() for x in collected_xs)
            _format_axes(ax, integer_x=integer_x)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(ncol=2)

            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / (
                f"step1_trajectories_{metric.lower()}_{filename_tag}_{algorithm}_{fixed_label}.png"
            )
            fig.tight_layout()
            fig.savefig(output, dpi=180)
            plt.close(fig)

    for algorithm in algorithms:
        algo_records = [
            r for r in records if str(r.get("algorithm") or "unknown") == algorithm
        ]
        node_values = sorted({r.get("num_nodes") for r in algo_records if r.get("num_nodes") is not None})
        period_values = sorted(
            {r.get("packet_interval_s") for r in algo_records if r.get("packet_interval_s") is not None}
        )
        if len(node_values) <= 1 and len(period_values) <= 1:
            continue

        for metric, ylabel in metrics.items():
            if len(node_values) > 1:
                plot_dimension(
                    algorithm=algorithm,
                    metric=metric,
                    ylabel=ylabel,
                    x_key="num_nodes",
                    fixed_key="packet_interval_s",
                    fixed_values=period_values,
                    x_label="Number of nodes",
                    filename_tag="nodes",
                )
            if len(period_values) > 1:
                plot_dimension(
                    algorithm=algorithm,
                    metric=metric,
                    ylabel=ylabel,
                    x_key="packet_interval_s",
                    fixed_key="num_nodes",
                    fixed_values=node_values,
                    x_label="Packet interval (s)",
                    filename_tag="interval",
                )


def _select_metric_value(record: Mapping[str, Any], metric: str) -> float:
    if metric == "snir_mean":
        value, _error_metric = _select_signal_mean(record)
        return _parse_float(value)
    return _parse_float(record.get(f"{metric}_mean")) or _parse_float(record.get(metric))


def _apply_ieee_style() -> None:
    if plt is None:
        return
    apply_plot_theme(plt)
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def _plot_snir_comparison(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return

    metrics = {
        "PDR": "Overall PDR",
        "DER": "Overall DER",
        "snir_mean": "Mean SNIR (dB)",
        "snr_mean": "Mean SNR (dB)",
        "collisions": "Collisions",
        "collisions_snir": "Collisions (SNIR)",
        "jain_index": "Jain index",
        "throughput_bps": "Aggregate throughput (bps)",
    }

    by_algorithm = defaultdict(list)
    for record in records:
        algo = str(record.get("algorithm") or "unknown")
        by_algorithm[algo].append(record)

    for algorithm, algo_records in sorted(by_algorithm.items()):
        periods = sorted({_parse_float(r.get("packet_interval_s")) for r in algo_records})
        for period in periods:
            period_records = [r for r in algo_records if _parse_float(r.get("packet_interval_s")) == period]
            if not period_records:
                continue
            for metric, ylabel in metrics.items():
                if not any(f"{metric}_mean" in r or metric in r for r in period_records):
                    continue
                error_metrics = {"PDR", "DER", "snir_mean", "snr_mean", "SNIR", "SNR"}

                def render(states: List[str], suffix: str, title: str) -> None:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    for state in states:
                        state_records = [
                            r
                            for r in period_records
                            if _record_matches_state(r, state)
                        ]
                        state_records.sort(key=lambda item: _parse_float(item.get("num_nodes")))
                        xs = [_parse_float(item.get("num_nodes")) for item in state_records]
                        ys = [_select_metric_value(item, metric) for item in state_records]
                        if not xs:
                            continue
                        label = _snir_label(state) if len(states) > 1 else _snir_label(state)
                        if metric in error_metrics:
                            lower: List[float] = []
                            upper: List[float] = []
                            for item, value in zip(state_records, ys):
                                error = _metric_error_bounds(item, metric, value)
                                if error is None:
                                    lower.append(float("nan"))
                                    upper.append(float("nan"))
                                else:
                                    low, high = error
                                    lower.append(low)
                                    upper.append(high)
                            has_errors = any(not (val != val) and val > 0 for val in lower + upper)
                            if has_errors:
                                ax.errorbar(
                                    xs,
                                    ys,
                                    yerr=[lower, upper],
                                    marker="o",
                                    markersize=6,
                                    linewidth=2,
                                    color=_snir_color(state),
                                    label=label,
                                    capsize=4,
                                )
                            else:
                                ax.plot(
                                    xs,
                                    ys,
                                    marker="o",
                                    markersize=6,
                                    linewidth=2,
                                    color=_snir_color(state),
                                    label=label,
                                )
                        else:
                            ax.plot(
                                xs,
                                ys,
                                marker="o",
                                markersize=6,
                                linewidth=2,
                                color=_snir_color(state),
                                label=label,
                            )

                    ax.set_xlabel("Number of nodes")
                    ax.set_ylabel(ylabel)
                    period_label = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
                    ax.set_title(f"{title} – {algorithm} – period {period_label} s")
                    _format_axes(ax, integer_x=True)
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend()

                    figures_dir.mkdir(parents=True, exist_ok=True)
                    output = figures_dir / (
                        f"algo_{algorithm}_{metric.lower()}_snir-compare{suffix}_tx_{period_label}.png"
                    )
                    fig.tight_layout()
                    fig.savefig(output, dpi=200)
                    plt.close(fig)

                _render_snir_variants(
                    render,
                    on_title=f"{ylabel} – SNIR enabled",
                    off_title=f"{ylabel} – SNIR disabled",
                    mixed_title=f"{ylabel} – mixed SNIR",
                )


def plot_distribution_by_state(
    records: List[Dict[str, Any]],
    figures_dir: Path,
    forced_algorithm: str | None = None,
) -> None:
    if not records or plt is None:
        return

    if forced_algorithm:
        records = _filter_records_for_algorithm(records, forced_algorithm)
        if not records:
            return

    metrics = {
        "snir_mean": "Mean SNIR (dB)",
        "DER": "Overall DER",
        "collisions": "Collisions",
    }
    states = ["snir_on", "snir_off"]

    def collect_values(state: str, metric: str) -> List[float]:
        values: List[float] = []
        for record in records:
            if not _record_matches_state(record, state):
                continue
            if metric == "snir_mean":
                value, _error_metric = _select_signal_mean(record)
            elif metric == "DER":
                value = record.get("DER") or record.get("DER_mean")
            else:
                value = record.get(metric)
            if value is None or value == "":
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return values

    for metric, ylabel in metrics.items():
        grouped = [collect_values(state, metric) for state in states]
        if not any(grouped):
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        boxplot = ax.boxplot(
            grouped,
            tick_labels=[_snir_label(state) for state in states],
            patch_artist=True,
            medianprops={"color": "#000000", "linewidth": 1.3},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, state in zip(boxplot["boxes"], states):
            patch.set_facecolor(_snir_color(state))
            patch.set_alpha(0.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("SNIR state")
        ax.set_title(f"{ylabel} distribution by SNIR state")
        _format_axes(ax, integer_x=False)
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = figures_dir / f"step1_distribution_{metric}.png"
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)


def generate_step1_figures(
    results_dir: Path,
    figures_dir: Path,
    use_summary: bool = False,
    plot_cdf: bool = False,
    plot_trajectories: bool = False,
    compare_snir: bool = True,
    strict: bool = False,
    official: bool = False,
    official_only: bool = False,
    ieee: bool = False,
) -> None:
    if plt is None:
        print("matplotlib n'est pas disponible ; aucune figure générée.")
        return

    _apply_ieee_style()
    if official and (not use_summary or not plot_cdf):
        raise ValueError(
            "Les figures officielles exigent --use-summary et --plot-cdf."
        )

    output_dir = figures_dir / "step1"
    extended_dir = output_dir / "extended"
    comparison_dir = extended_dir if official or official_only else output_dir
    trajectories_dir = extended_dir if use_summary or official or official_only else output_dir
    if official or official_only:
        output_dir = extended_dir
        extended_dir = output_dir
    comparison_records: List[Dict[str, Any]] = []

    if use_summary:
        summary_path = results_dir / "summary.csv"
        if (official or official_only) and not summary_path.exists():
            raise FileNotFoundError(
                f"summary.csv requis pour le mode IEEE, introuvable : {summary_path}"
            )
        summary_records = _load_summary_records(summary_path)
        if not summary_records:
            if official or official_only:
                raise FileNotFoundError(
                    f"Aucun enregistrement trouvé dans {summary_path} (mode IEEE)."
                )
            print(f"Aucun summary.csv trouvé dans {summary_path}; aucune barre générée.")
        else:
            if ieee:
                summary_records = _apply_ieee_filters(summary_records)
            _plot_summary_bars(
                summary_records,
                extended_dir,
                forced_algorithm=EXTENDED_ALGORITHM,
            )
            if plot_trajectories:
                trajectory_records = _load_step1_records(results_dir, strict=strict)
                if not trajectory_records:
                    trajectory_records = summary_records
                if ieee:
                    trajectory_records = _apply_ieee_filters(trajectory_records)
                _plot_trajectories(trajectory_records, trajectories_dir)
            comparison_records = summary_records
    elif not official_only:
        records = _load_step1_records(results_dir, strict=strict)
        if not records:
            print(f"Aucun CSV trouvé dans {results_dir} ; rien à tracer.")
            return
        if ieee:
            records = _apply_ieee_filters(records)
        _plot_cluster_der(records, output_dir)
        _plot_cluster_pdr(records, output_dir)
        _plot_global_metric(records, "PDR", "Overall PDR", "pdr_global", output_dir)
        _plot_global_metric(records, "DER", "Overall DER", "der_global", output_dir)
        _plot_global_metric(records, "collisions", "Collisions", "collisions", output_dir)
        _plot_global_metric(records, "collisions_snir", "Collisions (SNIR)", "collisions_snir", output_dir)
        _plot_global_metric(records, "jain_index", "Jain index", "jain_index", output_dir)
        _plot_global_metric(records, "throughput_bps", "Aggregate throughput (bps)", "throughput", output_dir)
        if any((r.get("snir_mean") is not None or r.get("snr_mean") is not None) for r in records):
            _plot_global_metric(records, "snir_mean", "Mean SNIR (dB)", "snir_mean", output_dir)
        if any(r.get("snr_mean") is not None for r in records):
            _plot_global_metric(records, "snr_mean", "Mean SNR (dB)", "snr_mean", output_dir)
        if plot_trajectories:
            _plot_trajectories(records, trajectories_dir)
        comparison_records = records

    if compare_snir:
        comparison_records = _load_comparison_records(results_dir, use_summary, strict)
        if ieee:
            comparison_records = _apply_ieee_filters(comparison_records)
        if not comparison_records:
            print("Aucune donnée disponible pour comparer SNIR on/off.")
        else:
            _plot_snir_comparison(comparison_records, comparison_dir)
            forced_algorithm = (
                EXTENDED_ALGORITHM if comparison_dir == extended_dir else None
            )
            plot_distribution_by_state(
                comparison_records,
                comparison_dir,
                forced_algorithm=forced_algorithm,
            )

    if plot_cdf:
        raw_path = results_dir / "raw_index.csv"
        if (official or official_only) and not raw_path.exists():
            raise FileNotFoundError(
                f"raw_index.csv requis pour le mode IEEE, introuvable : {raw_path}"
            )
        raw_records = _load_raw_samples(raw_path, results_dir, strict)
        if not raw_records:
            if official or official_only:
                raise FileNotFoundError(
                    f"Aucun échantillon brut trouvé dans {raw_path} (mode IEEE)."
                )
            print(f"Aucun échantillon brut trouvé dans {raw_path} ni dans {results_dir}.")
        else:
            if ieee:
                raw_records = _apply_ieee_filters(raw_records)
            _plot_cdf(
                raw_records,
                extended_dir,
                forced_algorithm=EXTENDED_ALGORITHM,
            )


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
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Utilise summary.csv pour tracer des barres avec intervalles de confiance",
    )
    parser.add_argument(
        "--plot-cdf",
        action="store_true",
        help="Active le tracé des CDF à partir de raw_index.csv ou des CSV bruts",
    )
    parser.add_argument(
        "--plot-trajectories",
        action="store_true",
        help="Ajoute les figures de trajectoires DER/PDR par seed (SNIR on/off superposés)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help=(
            "Génère les figures officielles dans figures/step1/extended/ "
            "(nécessite --use-summary et --plot-cdf)."
        ),
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help=(
            "Écrit uniquement dans figures/step1/extended/ (les figures standard ne sont pas recommandées)."
        ),
    )
    parser.add_argument(
        "--ieee",
        action="store_true",
        help=(
            "Applique un filtre de cohérence IEEE (SNIR on/off cohérents, snir_unknown exclus)."
        ),
    )
    parser.add_argument(
        "--compare-snir",
        action="store_true",
        default=True,
        help=(
            "Active les comparaisons SNIR on/off (activé par défaut, recommandé pour "
            "les figures 3×2 / 1×2 de README_FIGURES.md)."
        ),
    )
    parser.add_argument(
        "--no-compare-snir",
        action="store_false",
        dest="compare_snir",
        help="Désactive les figures combinées SNIR on/off",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Applique un filtrage strict des CSV (snir_state, snir_mean, snir_histogram_json) "
            "pour aligner la sélection sur les figures extended."
        ),
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.official and (not args.use_summary or not args.plot_cdf):
        parser.error("--official requiert --use-summary et --plot-cdf.")
    generate_step1_figures(
        args.results_dir,
        args.figures_dir,
        args.use_summary,
        args.plot_cdf,
        args.plot_trajectories,
        args.compare_snir,
        args.strict,
        args.official,
        args.official_only,
        args.ieee,
    )


if __name__ == "__main__":
    main()
