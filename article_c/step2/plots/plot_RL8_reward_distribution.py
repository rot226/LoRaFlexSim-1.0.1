"""Trace la figure RL8 (distribution des récompenses par algorithme)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    apply_figure_layout,
    CONSTANT_METRIC_MESSAGE,
    filter_rows_by_network_sizes,
    filter_cluster,
    is_constant_metric,
    normalize_network_size_rows,
    save_figure,
)
from plot_defaults import DEFAULT_FIGSIZE_MULTI


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
        if allow_sample:
            return []
        raise FileNotFoundError(f"CSV Step2 manquant: {results_path}")
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
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_MULTI)
    width, height = fig.get_size_inches()
    apply_figure_layout(fig, figsize=(width, height + 2))
    algorithms = sorted({row["algo"] for row in rows})
    rewards_by_algo = [
        [row["reward"] for row in rows if row["algo"] == algo]
        for algo in algorithms
    ]
    positions = list(range(1, len(algorithms) + 1))
    ax.violinplot(rewards_by_algo, positions=positions, showmedians=True)
    ax.boxplot(
        rewards_by_algo,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.6},
        medianprops={"color": "black"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([algo_label(str(algo)) for algo in algorithms])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Reward")
    ax.set_title(
        "Step 2 - Reward Distribution by Algorithm"
        f"{_title_suffix(network_sizes)}"
    )
    return fig


def _warn_if_constant(series: pd.Series, label: str) -> bool:
    if series.empty:
        warnings.warn(f"Aucune valeur disponible pour {label}.", stacklevel=2)
        return True
    values = [float(value) for value in series.dropna().tolist()]
    if is_constant_metric(values):
        warnings.warn(
            f"Valeurs constantes détectées pour {label} (variance faible).",
            stacklevel=2,
        )
        return True
    return False


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
    output_dir: Path,
    suffix: str,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2)
    apply_figure_layout(fig, figsize=(10, 10))
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

    rewards = pd.to_numeric(df.get("reward"), errors="coerce").dropna()
    axes[2].hist(rewards, bins="auto", color="#54a24b", alpha=0.8)
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
        axes[3].boxplot(rewards_by_algo, labels=[algo_label(str(a)) for a in algos])
        axes[3].set_title("Boxplot des récompenses par algo")
        axes[3].set_ylabel("Reward")
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Données algo absentes", ha="center", va="center")

    save_figure(fig, output_dir, f"{suffix}_diagnostics", use_tight=True)
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


def _plot_constant_message(
    message: str,
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_MULTI)
    apply_figure_layout(fig, figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(fig, output_dir, stem, use_tight=True)
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
    is_constant = _warn_if_constant(rewards_series, "reward")

    output_dir = step_dir / "plots" / "output"
    _plot_diagnostics(rows, output_dir, "plot_RL8_reward_distribution")
    _log_min_max_by_size(rows, "reward", label="reward")
    if is_constant:
        _plot_constant_message(
            CONSTANT_METRIC_MESSAGE,
            title="Step 2 - Reward Distribution by Algorithm",
            output_dir=output_dir,
            stem="plot_RL8_reward_distribution",
        )
        return
    fig = _plot_distribution(rows, network_sizes)
    save_figure(fig, output_dir, "plot_RL8_reward_distribution", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
