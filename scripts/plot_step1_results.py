"""Génère les figures de l'étape 1 à partir des CSV produits par Tâche 4."""

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

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures"

__all__ = ["generate_step1_figures", "DEFAULT_RESULTS_DIR", "DEFAULT_FIGURES_DIR"]

STATE_LABELS = {True: "snir_on", False: "snir_off", None: "snir_unknown"}
SNIR_COLORS = {"snir_on": "#d62728", "snir_off": "#1f77b4", "snir_unknown": "#7f7f7f"}
SNIR_LABELS = {
    "snir_on": "SNIR activé",
    "snir_off": "SNIR désactivé",
    "snir_unknown": "SNIR inconnu",
}
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


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

    for key in ("use_snir", "with_snir"):
        parsed = _parse_bool(row.get(key))
        if parsed is not None:
            return STATE_LABELS.get(parsed, "snir_unknown"), True
    return None, False


def _record_matches_state(record: Mapping[str, Any], state: str) -> bool:
    return record.get("snir_state") == state and record.get("snir_detected", True)


def _load_step1_records(results_dir: Path, strict: bool = False) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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
                for key, value in row.items():
                    if key.startswith("qos_cluster_pdr__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_pdr[cluster_id] = _parse_float(value)
                    elif key.startswith("qos_cluster_targets__"):
                        cluster_id = int(key.split("__")[-1])
                        cluster_targets[cluster_id] = _parse_float(value)

                snir_candidate = row.get("snir_mean")
                snr_candidate = row.get("snr_mean") or row.get("SNR") or row.get("snr")
                record: Dict[str, Any] = {
                    "csv_path": csv_path,
                    "algorithm": row.get("algorithm", csv_path.parent.name),
                    "num_nodes": int(float(row.get("num_nodes", "0") or 0)),
                    "packet_interval_s": float(row.get("packet_interval_s", "0") or 0),
                    "PDR": _parse_float(row.get("PDR")),
                    "DER": _parse_float(row.get("DER")),
                    "snir_mean": _maybe_float(snir_candidate),
                    "snr_mean": _maybe_float(snr_candidate),
                    "collisions": int(float(row.get("collisions", "0") or 0)),
                    "collisions_snir": int(float(row.get("collisions_snir", "0") or 0)),
                    "jain_index": _parse_float(row.get("jain_index")),
                    "throughput_bps": _parse_float(row.get("throughput_bps")),
                    "cluster_pdr": cluster_pdr,
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
                records.append(record)
    return records


def _snir_label(state: str | None) -> str:
    return SNIR_LABELS.get(state or "snir_unknown", SNIR_LABELS["snir_unknown"])


def _snir_color(state: str | None) -> str:
    return SNIR_COLORS.get(state or "snir_unknown", SNIR_COLORS["snir_unknown"])


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
        render(states, suffix, title)


def _format_axes(ax: Any, integer_x: bool = False) -> None:
    if plt is None:
        return
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.5)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="both", direction="in", length=4.5, width=1.0, labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.title.set_size(12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for line in ax.get_lines():
        line.set_linewidth(2.0)
        line.set_markersize(6.0)
        line.set_markeredgewidth(0.8)


def _metric_error_bounds(record: Mapping[str, Any], metric: str, value: float) -> Tuple[float, float] | None:
    for base in {metric, metric.lower(), metric.upper()}:
        ci_low = _maybe_float(record.get(f"{base}_ci_low"))
        ci_high = _maybe_float(record.get(f"{base}_ci_high"))
        if ci_low is not None and ci_high is not None:
            return max(0.0, value - ci_low), max(0.0, ci_high - value)
        std = _maybe_float(record.get(f"{base}_std"))
        if std is not None:
            return float(std), float(std)
    return None


def _load_summary_records(summary_path: Path, forced_state: str | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not summary_path.exists():
        return records

    with summary_path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record: Dict[str, Any] = {"summary_path": summary_path}
            for key, value in row.items():
                if key in {"algorithm", "snir_state"}:
                    record[key] = value
                elif key in {"num_nodes"}:
                    record[key] = int(float(value or 0))
                elif key in {"packet_interval_s"}:
                    record[key] = float(value or 0)
                else:
                    record[key] = _parse_float(value)
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
            record["snir_detected"] = snir_detected
            records.append(record)
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
                record: Dict[str, Any] = {
                    "algorithm": row.get("algorithm"),
                    "snir_state": row.get("snir_state", "snir_unknown"),
                    "packet_interval_s": _parse_float(row.get("packet_interval_s")),
                    "DER": _parse_float(row.get("DER")),
                }
                records.append(record)
    else:
        records = _load_step1_records(fallback_dir, strict=strict)
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
                    for item in data:
                        raw_value = item.get(metric)
                        if raw_value is None:
                            continue
                        xs.append(item["num_nodes"])
                        ys.append(_parse_float(raw_value))
                        used_items.append(item)
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
                        for item, value in zip(used_items, ys):
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
            ax.set_xlabel("Nombre de nœuds")
            ax.set_ylabel(ylabel)
            title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            ax.set_title(f"{title} – période {title_period} s")
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
        mixed_title=f"{ylabel} – SNIR mixte",
    )


def _plot_summary_bars(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return

    metrics = {
        "PDR": "PDR global",
        "DER": "DER global",
        "snir_mean": "SNIR moyen (dB)",
        "snr_mean": "SNR moyen (dB)",
        "collisions": "Collisions",
        "collisions_snir": "Collisions (SNIR)",
        "jain_index": "Indice de Jain",
        "throughput_bps": "Débit agrégé (bps)",
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
                    values.append(match.get(f"{metric}_mean", 0.0) if match else 0.0)
                    errors.append(match.get(f"{metric}_std", 0.0) if match else 0.0)

                ax.bar(
                    offsets,
                    values,
                    width=width,
                    yerr=errors,
                    label=_snir_label(state),
                    color=_snir_color(state),
                    capsize=4,
                    edgecolor="black",
                    linewidth=0.9,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{algo}\n{nodes} nœuds" for nodes, algo in combinations], rotation=0)
            ax.set_ylabel(ylabel)
            period_label = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            ax.set_title(f"{ylabel} – période {period_label} s")
            _format_axes(ax, integer_x=False)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"summary_{metric.lower()}_tx_{period_label}.png"
            fig.tight_layout()
            fig.savefig(output, dpi=150)
            plt.close(fig)


def _plot_cdf(records: Sequence[Mapping[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
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
                label=_snir_label(state),
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
    algorithms = sorted({r["algorithm"] for r in records})
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
                    ax.axhline(target, color="black", linestyle="--", linewidth=1, label="Cible" if idx == 0 else None)
                ax.set_title(f"Cluster {cluster_id}")
                ax.set_xlabel("Nœuds")
                if idx == 0:
                    ax.set_ylabel("PDR")
                ax.set_ylim(0.0, 1.05)
                _format_axes(ax, integer_x=True)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
            title_period = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            fig.suptitle(f"{title} – période {title_period} s")
            figures_dir.mkdir(parents=True, exist_ok=True)
            output = figures_dir / f"step1_cluster_pdr{suffix}_tx_{title_period}.png"
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(output, dpi=150)
            plt.close(fig)

    _render_snir_variants(
        render,
        on_title="PDR par cluster – SNIR activé",
        off_title="PDR par cluster – SNIR désactivé",
        mixed_title="PDR par cluster – SNIR mixte",
    )


def _select_metric_value(record: Mapping[str, Any], metric: str) -> float:
    return _parse_float(record.get(f"{metric}_mean")) or _parse_float(record.get(metric))


def _apply_ieee_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 200,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def _plot_snir_comparison(records: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not records or plt is None:
        return

    metrics = {
        "PDR": "PDR global",
        "DER": "DER global",
        "snir_mean": "SNIR moyen (dB)",
        "snr_mean": "SNR moyen (dB)",
        "collisions": "Collisions",
        "collisions_snir": "Collisions (SNIR)",
        "jain_index": "Indice de Jain",
        "throughput_bps": "Débit agrégé (bps)",
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

                    ax.set_xlabel("Nombre de nœuds")
                    ax.set_ylabel(ylabel)
                    period_label = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
                    ax.set_title(f"{title} – {algorithm} – période {period_label} s")
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
                    on_title=f"{ylabel} – SNIR activé",
                    off_title=f"{ylabel} – SNIR désactivé",
                    mixed_title=f"{ylabel} – SNIR mixte",
                )


def generate_step1_figures(
    results_dir: Path,
    figures_dir: Path,
    use_summary: bool = False,
    plot_cdf: bool = False,
    compare_snir: bool = True,
    strict: bool = False,
    official: bool = False,
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
    comparison_dir = extended_dir if official else output_dir
    if official:
        output_dir = extended_dir
        extended_dir = output_dir
    comparison_records: List[Dict[str, Any]] = []

    if use_summary:
        summary_path = results_dir / "summary.csv"
        summary_records = _load_summary_records(summary_path)
        if not summary_records:
            print(f"Aucun summary.csv trouvé dans {summary_path}; aucune barre générée.")
        else:
            _plot_summary_bars(summary_records, extended_dir)
            comparison_records = summary_records
    else:
        records = _load_step1_records(results_dir, strict=strict)
        if not records:
            print(f"Aucun CSV trouvé dans {results_dir} ; rien à tracer.")
            return
        _plot_cluster_pdr(records, output_dir)
        _plot_global_metric(records, "PDR", "PDR global", "pdr_global", output_dir)
        _plot_global_metric(records, "DER", "DER global", "der_global", output_dir)
        _plot_global_metric(records, "collisions", "Collisions", "collisions", output_dir)
        _plot_global_metric(records, "collisions_snir", "Collisions (SNIR)", "collisions_snir", output_dir)
        _plot_global_metric(records, "jain_index", "Indice de Jain", "jain_index", output_dir)
        _plot_global_metric(records, "throughput_bps", "Débit agrégé (bps)", "throughput", output_dir)
        if any(r.get("snir_mean") is not None for r in records):
            _plot_global_metric(records, "snir_mean", "SNIR moyen (dB)", "snir_mean", output_dir)
        if any(r.get("snr_mean") is not None for r in records):
            _plot_global_metric(records, "snr_mean", "SNR moyen (dB)", "snr_mean", output_dir)
        comparison_records = records

    if compare_snir:
        comparison_records = _load_comparison_records(results_dir, use_summary, strict)
        if not comparison_records:
            print("Aucune donnée disponible pour comparer SNIR on/off.")
        else:
            _plot_snir_comparison(comparison_records, comparison_dir)

    if plot_cdf:
        raw_path = results_dir / "raw_index.csv"
        raw_records = _load_raw_samples(raw_path, results_dir, strict)
        if not raw_records:
            print(f"Aucun échantillon brut trouvé dans {raw_path} ni dans {results_dir}.")
        else:
            _plot_cdf(raw_records, extended_dir)


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
        "--official",
        action="store_true",
        help=(
            "Génère les figures officielles dans figures/step1/extended/ "
            "(nécessite --use-summary et --plot-cdf)."
        ),
    )
    parser.add_argument(
        "--compare-snir",
        action="store_true",
        default=True,
        help=(
            "Active les figures combinées SNIR on/off par métrique et par algorithme (activé par défaut)"
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
        args.compare_snir,
        args.strict,
        args.official,
    )


if __name__ == "__main__":
    main()
