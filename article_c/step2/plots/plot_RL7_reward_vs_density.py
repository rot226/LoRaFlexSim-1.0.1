"""Trace la figure RL7 (récompense médiane globale vs densité)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    load_step2_aggregated,
    normalize_network_size_rows,
    place_legend,
    plot_metric_by_algo,
    resolve_percentile_keys,
    save_figure,
)

ALGO_ALIASES = {
    "adr": "adr",
    "ADR": "adr",
    "mixra_h": "mixra_h",
    "MixRA-H": "mixra_h",
    "mixra_opt": "mixra_opt",
    "MixRA-Opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "UCB1-SF": "ucb1_sf",
}
TARGET_ALGOS = {"adr", "mixra_h", "mixra_opt", "ucb1_sf"}


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


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _label_for_algo(algo: str) -> str:
    canonical = _canonical_algo(algo)
    if canonical is None:
        return algo
    return algo_label(canonical)


def _filter_algorithms(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = [
        row for row in rows if _canonical_algo(str(row.get("algo", ""))) in TARGET_ALGOS
    ]
    return filtered or rows


def _plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int] | None,
) -> plt.Figure | None:
    fig, ax = plt.subplots()
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    if network_sizes is None:
        network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    plot_metric_by_algo(
        ax,
        rows,
        metric_key,
        network_sizes,
        label_fn=lambda algo: _label_for_algo(str(algo)),
    )
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Median Reward (global, p10-p90)")
    ax.set_title(
        "Step 2 - Global Median Reward vs Network size (adaptive algorithms)"
        f"{_title_suffix(network_sizes)}"
    )
    place_legend(ax)
    return fig


def _extract_metric_values(
    rows: list[dict[str, object]],
    metric_key: str,
) -> tuple[pd.Series, str]:
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    values = pd.Series(
        [
            row.get(median_key)
            for row in rows
            if isinstance(row.get(median_key), (int, float))
        ]
    )
    return values, median_key


def _warn_if_constant(series: pd.Series, label: str) -> None:
    if series.empty:
        warnings.warn(f"Aucune valeur disponible pour {label}.", stacklevel=2)
        return
    if series.nunique(dropna=True) <= 1:
        warnings.warn(
            f"Valeurs constantes détectées pour {label} (variance nulle).",
            stacklevel=2,
        )


def _diagnose_density(rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    if "density" not in df.columns:
        warnings.warn("Colonne density absente: impossible de valider la densité.", stacklevel=2)
        return
    density = pd.to_numeric(df["density"], errors="coerce").dropna()
    _warn_if_constant(density, "density")
    if "network_size" in df.columns:
        network_size = pd.to_numeric(df["network_size"], errors="coerce").dropna()
        aligned = pd.concat([network_size, density], axis=1).dropna()
        if not aligned.empty:
            area = aligned.iloc[:, 0] / aligned.iloc[:, 1].replace(0, pd.NA)
            area = area.dropna()
            _warn_if_constant(area, "area (network_size / density)")


def _plot_diagnostics(
    rows: list[dict[str, object]],
    metric_key: str,
    output_dir: Path,
    suffix: str,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    metric_values, metric_label = _extract_metric_values(rows, metric_key)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    network_sizes = pd.to_numeric(df.get("network_size"), errors="coerce").dropna()
    axes[0].hist(network_sizes, bins="auto", color="#4c78a8", alpha=0.8)
    axes[0].set_title("Histogramme des tailles de réseau")
    axes[0].set_xlabel("Network size")

    if "density" in df.columns:
        density = pd.to_numeric(df["density"], errors="coerce").dropna()
        axes[1].hist(density, bins="auto", color="#f58518", alpha=0.8)
        axes[1].set_title("Histogramme des densités")
        axes[1].set_xlabel("Density")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Density absente", ha="center", va="center")

    axes[2].hist(metric_values, bins="auto", color="#54a24b", alpha=0.8)
    axes[2].set_title(f"Histogramme {metric_label}")
    axes[2].set_xlabel(metric_label)

    if "algo" in df.columns and not metric_values.empty:
        algos = sorted(df["algo"].dropna().unique())
        rewards_by_algo = [
            [
                row.get(metric_label)
                for row in rows
                if row.get("algo") == algo and isinstance(row.get(metric_label), (int, float))
            ]
            for algo in algos
        ]
        axes[3].boxplot(rewards_by_algo, labels=[_label_for_algo(str(a)) for a in algos])
        axes[3].set_title("Boxplot des récompenses par algo")
        axes[3].set_ylabel(metric_label)
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Données algo absentes", ha="center", va="center")

    save_figure(fig, output_dir, f"{suffix}_diagnostics", use_tight=True)
    plt.close(fig)


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)
    rows = _filter_algorithms(rows)
    _diagnose_density(rows)
    metric_values, metric_label = _extract_metric_values(rows, "reward_mean")
    _warn_if_constant(metric_values, metric_label)

    output_dir = step_dir / "plots" / "output"
    _plot_diagnostics(rows, "reward_mean", output_dir, "plot_RL7_reward_vs_density")
    fig = _plot_metric(rows, "reward_mean", network_sizes_filter)
    if fig is None:
        return
    save_figure(fig, output_dir, "plot_RL7_reward_vs_density", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
