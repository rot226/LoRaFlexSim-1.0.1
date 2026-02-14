"""Trace la figure RL5+ (résumé par percentile)."""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    assert_legend_present,
    MetricStatus,
    fallback_legend_handles,
    filter_rows_by_network_sizes,
    is_constant_metric,
    load_step2_aggregated,
    load_step2_selection_probs,
    normalize_network_size_rows,
    place_adaptive_legend,
    render_metric_status,
    save_figure,
    warn_metric_checks,
)
from article_c.common.plot_data_source import load_aggregated_rows_for_source
from plot_defaults import RL_FIGURE_SCALE, resolve_ieee_figsize


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes:
        return None
    return network_sizes


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        print(
            "ERREUR: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée."
        )
        return True
    return False


def _title_suffix(network_sizes: list[int]) -> str:
    if len(network_sizes) == 1:
        return " (taille unique)"
    return ""


def _plot_selection(
    rows: list[dict[str, object]],
    network_sizes: list[int],
    *,
    lite: bool,
) -> plt.Figure:
    network_sizes = sorted(network_sizes)
    sfs = sorted({row["sf"] for row in rows})
    series_count = len(sfs) * len(network_sizes) if sfs and network_sizes else None
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count, scale=RL_FIGURE_SCALE))
    selection_values = [
        float(row.get("selection_prob"))
        for row in rows
        if isinstance(row.get("selection_prob"), (int, float))
    ]
    warn_metric_checks(
        selection_values,
        "Probabilité de sélection",
        min_value=0.0,
        max_value=1.0,
    )
    for network_size in network_sizes:
        size_rows = [row for row in rows if row["network_size"] == network_size]
        ordered_probs = [
            float(row.get("selection_prob"))
            for row in sorted(size_rows, key=lambda item: item.get("sf"))
            if isinstance(row.get("selection_prob"), (int, float))
        ]
        cumulative = 0.0
        cdf_values = []
        for value in ordered_probs:
            cumulative += value
            cdf_values.append(cumulative)
        warn_metric_checks(
            cdf_values,
            f"CDF sélection SF (N={network_size})",
            min_value=0.0,
            max_value=1.0,
            expected_monotonic="nondecreasing",
        )
    metric_state = is_constant_metric(selection_values)
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            ax,
            metric_state,
            legend_loc="right",
            show_fallback_legend=True,
            legend_handles=fallback_legend_handles(),
        )
        return fig
    data = _summarize_selection(rows, network_sizes)
    max_points = 60 if lite else None
    for network_size in network_sizes:
        for sf in sfs:
            summary = data[
                (data["network_size"] == network_size) & (data["sf"] == sf)
            ].sort_values("round")
            if summary.empty:
                continue
            rounds = summary["round"].tolist()
            p10 = summary["p10"].tolist()
            p50 = summary["p50"].tolist()
            p90 = summary["p90"].tolist()
            if max_points is not None:
                indices = _downsample_indices(len(rounds), max_points)
                rounds = [rounds[i] for i in indices]
                p10 = [p10[i] for i in indices]
                p50 = [p50[i] for i in indices]
                p90 = [p90[i] for i in indices]
            label = f"SF {sf} (N={network_size})"
            ax.plot(rounds, p50, label=label)
            ax.fill_between(rounds, p10, p90, alpha=0.15, label=f"{label} (P10–P90)")
    ax.set_xlabel("Round (index)")
    ax.set_ylabel("Selection prob. (prob.)")
    place_adaptive_legend(fig, ax, preferred_loc="right")
    return fig


def _summarize_selection(
    rows: list[dict[str, object]],
    network_sizes: list[int],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "network_size" not in df.columns:
        if len(network_sizes) == 1:
            df["network_size"] = network_sizes[0]
        else:
            df["network_size"] = 0
    grouped = df.groupby(["network_size", "sf", "round"])["selection_prob"]
    summary = grouped.agg(
        p10=lambda values: values.quantile(0.1),
        p50=lambda values: values.quantile(0.5),
        p90=lambda values: values.quantile(0.9),
    )
    return summary.reset_index()


def _downsample_indices(length: int, max_points: int) -> list[int]:
    if max_points <= 0 or length <= max_points:
        return list(range(length))
    if max_points == 1:
        return [0]
    last_index = length - 1
    indices: list[int] = []
    for idx in range(max_points):
        position = int(round(idx * last_index / (max_points - 1)))
        if not indices or position != indices[-1]:
            indices.append(position)
    return indices


def main(
    network_sizes: list[int] | None = None,
            argv: list[str] | None = None,
    allow_sample: bool = True,
    lite: bool = False, source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Active un mode compact (downsample des points).",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if lite is False:
        lite = args.lite
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "rl5_selection_prob.csv"
    if not allow_sample and not results_path.exists():
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = load_step2_selection_probs(results_path)
    size_rows = load_aggregated_rows_for_source(
        step_dir=step_dir,
        source=LAST_EFFECTIVE_SOURCE,
        step_label="Step2",
        loader=load_step2_aggregated,
        allow_sample=allow_sample,
    )
    if not size_rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    normalize_network_size_rows(size_rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    size_rows, _ = filter_rows_by_network_sizes(size_rows, network_sizes_filter)
    if network_sizes_filter is None:
        df = pd.DataFrame(size_rows)
        network_sizes = sorted(df["network_size"].unique())
    else:
        network_sizes = network_sizes_filter
    if _has_invalid_network_sizes(network_sizes):
        return
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )

    rows, _ = filter_rows_by_network_sizes(rows, network_sizes)
    fig = _plot_selection(rows, network_sizes, lite=lite)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL5_plus", use_tight=False)
    assert_legend_present(fig, "plot_RL5_plus")
    plt.close(fig)


if __name__ == "__main__":
    main()
