"""Trace la figure RL8 (distribution des récompenses par algorithme)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    apply_figure_layout,
    ALGO_COLORS,
    MetricStatus,
    assert_legend_present,
    filter_rows_by_network_sizes,
    filter_cluster,
    is_constant_metric,
    normalize_network_size_rows,
    add_global_legend,
    save_figure,
)
from article_c.common.plotting_style import legend_extra_height
from plot_defaults import resolve_ieee_figsize


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


def _load_step2_raw_results(
    results_path: Path,
    *,
    allow_sample: bool = True,
) -> list[dict[str, object]]:
    if not results_path.exists():
        warnings.warn(f"CSV Step2 manquant: {results_path}", stacklevel=2)
        return []
    df = pd.read_csv(results_path)
    if df.empty:
        return []
    if "reward" not in df.columns:
        warnings.warn(
            f"Colonne reward manquante dans {results_path.name}; figure ignorée.",
            stacklevel=2,
        )
        return []
    if "network_size" in df.columns:
        network_size_series = df["network_size"]
    elif "density" in df.columns:
        network_size_series = df["density"]
    else:
        network_size_series = pd.Series([None] * len(df))
    df["network_size"] = pd.to_numeric(network_size_series, errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    if "density" in df.columns:
        df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["algo"] = df.get("algo", "")
    df["snir_mode"] = df.get("snir_mode", "")
    df["cluster"] = df.get("cluster", "all").fillna("all")
    df = df.dropna(subset=["network_size", "reward"])
    return df.to_dict(orient="records")


def _plot_distribution(
    rows: list[dict[str, object]],
    network_sizes: list[int],
) -> plt.Figure:
    algorithms = sorted({row["algo"] for row in rows})
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    width, height = fig.get_size_inches()
    algo_colors = [ALGO_COLORS.get(str(algo), "#333333") for algo in algorithms]
    rewards_by_algo = [
        [row["reward"] for row in rows if row["algo"] == algo]
        for algo in algorithms
    ]
    positions = list(range(1, len(algorithms) + 1))
    violin_parts = ax.violinplot(
        rewards_by_algo, positions=positions, showmedians=True
    )
    for body, color, algo in zip(
        violin_parts["bodies"],
        algo_colors,
        algorithms,
        strict=False,
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
        body.set_label(f"{algo_label(str(algo))} (violon)")
    boxplot_parts = ax.boxplot(
        rewards_by_algo,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.6},
        medianprops={"color": "black"},
    )
    for patch, algo in zip(boxplot_parts.get("boxes", []), algorithms, strict=False):
        patch.set_label(f"{algo_label(str(algo))} (boxplot)")
    ax.set_xticks(positions)
    ax.set_xticklabels([algo_label(str(algo)) for algo in algorithms])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Reward")
    ax.set_title(
        "Step 2 - Reward Distribution by Algorithm"
        f"{_title_suffix(network_sizes)}"
    )
    legend_rows = (
        max(1, math.ceil(len(algorithms) / min(3, len(algorithms))))
        if algorithms
        else 1
    )
    extra_height = legend_extra_height(height, legend_rows, legend_loc="right")
    apply_figure_layout(fig, figsize=(width, height + extra_height))
    add_global_legend(fig, ax, legend_loc="right")
    return fig


def _metric_status(series: pd.Series, label: str) -> MetricStatus:
    if series.empty:
        warnings.warn(f"Aucune valeur disponible pour {label} (données absentes).", stacklevel=2)
        return MetricStatus.MISSING
    values = [float(value) for value in series.dropna().tolist()]
    if not values:
        warnings.warn(f"Aucune valeur disponible pour {label} (données absentes).", stacklevel=2)
        return MetricStatus.MISSING
    metric_state = is_constant_metric(values)
    if metric_state is MetricStatus.CONSTANT:
        warnings.warn(
            f"Valeurs constantes détectées pour {label} (variance faible).",
            stacklevel=2,
        )
        return MetricStatus.CONSTANT
    return MetricStatus.OK


def _emit_step2_tuning_message() -> None:
    message = (
        "Step2: ajustez les paramètres trafic, collision, link_quality pour "
        "obtenir une récompense exploitable."
    )
    warnings.warn(message, stacklevel=2)
    print(f"INFO: {message}")


def _diagnose_density(rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    if "density" not in df.columns:
        warnings.warn("Colonne density absente: impossible de valider la densité.", stacklevel=2)
        return
    density = pd.to_numeric(df["density"], errors="coerce").dropna()
    density_status = _metric_status(density, "density")
    if "network_size" in df.columns:
        network_size = pd.to_numeric(df["network_size"], errors="coerce").dropna()
        aligned = pd.concat([network_size, density], axis=1).dropna()
        if not aligned.empty:
            area = aligned.iloc[:, 0] / aligned.iloc[:, 1].replace(0, pd.NA)
            area = area.dropna()
            if density_status is MetricStatus.CONSTANT:
                _metric_status(area, "area (network_size / density)")


def _plot_diagnostics(
    rows: list[dict[str, object]],
    output_dir: Path,
    suffix: str,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    base_width, base_height = resolve_ieee_figsize(2)
    figsize = (base_width, base_height * 2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    apply_figure_layout(fig, figsize=figsize)
    axes = axes.flatten()

    network_sizes = pd.to_numeric(df.get("network_size"), errors="coerce").dropna()
    axes[0].hist(
        network_sizes,
        bins="auto",
        color="#4c78a8",
        alpha=0.8,
        label="Network size",
    )
    axes[0].set_title("Histogramme des tailles de réseau")
    axes[0].set_xlabel("Network size")

    if "density" in df.columns:
        density = pd.to_numeric(df["density"], errors="coerce").dropna()
        axes[1].hist(
            density,
            bins="auto",
            color="#f58518",
            alpha=0.8,
            label="Density",
        )
        axes[1].set_title("Histogramme des densités")
        axes[1].set_xlabel("Density")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Density absente", ha="center", va="center")

    rewards = pd.to_numeric(df.get("reward"), errors="coerce").dropna()
    axes[2].hist(
        rewards,
        bins="auto",
        color="#54a24b",
        alpha=0.8,
        label="Reward",
    )
    axes[2].set_title("Histogramme des récompenses")
    axes[2].set_xlabel("Reward")

    if "algo" in df.columns and not rewards.empty:
        algos = sorted(df["algo"].dropna().unique())
        rewards_by_algo = [
            [
                row.get("reward")
                for row in rows
                if row.get("algo") == algo and isinstance(row.get("reward"), (int, float))
            ]
            for algo in algos
        ]
        boxplot_parts = axes[3].boxplot(
            rewards_by_algo,
            labels=[algo_label(str(a)) for a in algos],
            patch_artist=True,
        )
        for patch, algo in zip(boxplot_parts.get("boxes", []), algos, strict=False):
            patch.set_label(f"{algo_label(str(algo))} (boxplot)")
        axes[3].set_title("Boxplot des récompenses par algo")
        axes[3].set_ylabel("Reward")
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Données algo absentes", ha="center", va="center")

    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in axes:
        subplot_handles, subplot_labels = ax.get_legend_handles_labels()
        for handle, label in zip(subplot_handles, subplot_labels, strict=False):
            if label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    if handles:
        add_global_legend(
            fig,
            axes[0],
            legend_loc="right",
            handles=handles,
            labels=labels,
            use_fallback=False,
        )
    save_figure(fig, output_dir, f"{suffix}_diagnostics", use_tight=False)
    assert_legend_present(fig, f"{suffix}_diagnostics")
    plt.close(fig)


def _log_min_max_by_size(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    label: str,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty or "network_size" not in df.columns:
        warnings.warn("Diagnostic min/max indisponible: données absentes.", stacklevel=2)
        return
    values = pd.to_numeric(df.get(metric_key), errors="coerce")
    df = pd.DataFrame(
        {"network_size": pd.to_numeric(df.get("network_size"), errors="coerce"), "value": values}
    ).dropna()
    if df.empty:
        warnings.warn(
            f"Diagnostic min/max indisponible pour {label}: valeurs absentes.",
            stacklevel=2,
        )
        return
    print(f"Diagnostic min/max pour {label} (par taille):")
    grouped = df.groupby("network_size")["value"]
    for size, stats in grouped.agg(["min", "max"]).sort_index().iterrows():
        print(f"  taille={int(size)} -> min={stats['min']:.6f} / max={stats['max']:.6f}")


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
    results_path = step_dir / "results" / "raw_all.csv"
    rows = _load_step2_raw_results(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)
    if network_sizes_filter is None:
        df = pd.DataFrame(rows)
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

    _diagnose_density(rows)
    rewards_series = pd.to_numeric(
        pd.Series([row.get("reward") for row in rows]), errors="coerce"
    ).dropna()
    metric_status = _metric_status(rewards_series, "reward")
    output_dir = step_dir / "plots" / "output"
    if metric_status is MetricStatus.MISSING:
        warnings.warn("Données reward absentes: figure ignorée.", stacklevel=2)
        _emit_step2_tuning_message()
        return
    if metric_status is MetricStatus.CONSTANT:
        warnings.warn("reward constant → plots invalides", stacklevel=2)
        print("INFO: reward constant; génération d'un diagnostic uniquement.")
        _plot_diagnostics(rows, output_dir, "plot_RL8_reward_distribution")
        _emit_step2_tuning_message()
        return
    _plot_diagnostics(rows, output_dir, "plot_RL8_reward_distribution")
    _log_min_max_by_size(rows, "reward", label="reward")
    fig = _plot_distribution(rows, network_sizes)
    save_figure(fig, output_dir, "plot_RL8_reward_distribution", use_tight=False)
    assert_legend_present(fig, "plot_RL8_reward_distribution")
    plt.close(fig)


if __name__ == "__main__":
    main()
