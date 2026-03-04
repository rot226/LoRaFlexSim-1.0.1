"""CLI de génération des figures de campagne SFRD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_REQUIRED_FILES = {
    "SNIR_OFF/pdr_results.csv": "pdr_results.csv",
    "SNIR_OFF/throughput_results.csv": "throughput_results.csv",
    "SNIR_OFF/energy_results.csv": "energy_results.csv",
    "SNIR_OFF/sf_distribution.csv": "sf_distribution.csv",
    "SNIR_ON/pdr_results.csv": "pdr_results.csv",
    "SNIR_ON/throughput_results.csv": "throughput_results.csv",
    "SNIR_ON/energy_results.csv": "energy_results.csv",
    "SNIR_ON/sf_distribution.csv": "sf_distribution.csv",
    "learning_curve_ucb.csv": "learning_curve_ucb.csv",
}

_FIXED_FIGURE_SPECS = (
    ("pdr_vs_n", "SNIR_OFF/pdr_results.csv", "SNIR_ON/pdr_results.csv"),
    (
        "throughput_vs_n",
        "SNIR_OFF/throughput_results.csv",
        "SNIR_ON/throughput_results.csv",
    ),
    ("energy_vs_n", "SNIR_OFF/energy_results.csv", "SNIR_ON/energy_results.csv"),
    ("sf_distribution", "SNIR_OFF/sf_distribution.csv", "SNIR_ON/sf_distribution.csv"),
    ("learning_curve_ucb", "learning_curve_ucb.csv"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Génère les figures article (PDR/Throughput/Energy vs N, "
            "distribution SF et learning curve UCB) depuis un dossier de campagne."
        )
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Identifiant de campagne sous sfrd/logs/<campaign_id>.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Racine des campagnes (défaut: sfrd/logs).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Racine contenant les CSV agrégés (SNIR_OFF/, SNIR_ON/, learning_curve_ucb.csv). "
            "Si omis, déduit de --campaign-id."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Dossier de sortie des figures. Défaut: figures/<campaign_id>/.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Format des figures (défaut: png).",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_root is not None:
        output_root = args.output_root.resolve()
        if args.figures_dir is not None:
            figures_dir = args.figures_dir.resolve()
        else:
            figures_dir = output_root.parent / "figures"
        return output_root, figures_dir

    if not args.campaign_id:
        raise ValueError("Fournir --campaign-id ou --output-root.")

    campaign_root = (args.logs_root / args.campaign_id).resolve()
    output_root = campaign_root / "output"
    if args.figures_dir is not None:
        figures_dir = args.figures_dir.resolve()
    else:
        figures_dir = Path("figures") / args.campaign_id
    return output_root, figures_dir


def _write_figures_manifest(
    *,
    manifest_path: Path,
    campaign_id: str | None,
    output_root: Path,
    figures_dir: Path,
    format_name: str,
) -> None:
    figures = []
    for spec in _FIXED_FIGURE_SPECS:
        figure_name = spec[0]
        source_csvs = [str((output_root / rel).resolve()) for rel in spec[1:]]
        figures.append(
            {
                "name": figure_name,
                "file": str((figures_dir / f"{figure_name}.{format_name}").resolve()),
                "source_csv": source_csvs,
            }
        )

    payload = {
        "campaign_id": campaign_id,
        "output_root": str(output_root.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "format": format_name,
        "figures": figures,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _load_required_csvs(output_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for relative_path in _REQUIRED_FILES:
        csv_path = output_root / relative_path
        if not csv_path.is_file():
            missing.append(str(csv_path))
            continue
        frames[relative_path] = pd.read_csv(csv_path)

    if missing:
        raise FileNotFoundError(
            "Fichiers requis manquants pour le plotting:\n- " + "\n- ".join(missing)
        )
    return frames


def _plot_metric_vs_n(
    off_df: pd.DataFrame,
    on_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for snir_label, df in (("OFF", off_df), ("ON", on_df)):
        for algo, subset in df.groupby("algorithm"):
            sorted_subset = subset.sort_values("network_size")
            ax.plot(
                sorted_subset["network_size"],
                sorted_subset[metric_col],
                marker="o",
                linewidth=1.8,
                label=f"{algo} ({snir_label})",
            )

    ax.set_xlabel("Nombre de nœuds (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_sf_distribution(
    off_df: pd.DataFrame,
    on_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (snir_label, df) in zip(axes, (("OFF", off_df), ("ON", on_df))):
        grouped = (
            df.groupby(["algorithm", "sf"], as_index=False)["count"]
            .mean()
            .sort_values(["algorithm", "sf"])
        )

        for algo, subset in grouped.groupby("algorithm"):
            ax.plot(subset["sf"], subset["count"], marker="o", label=algo)

        ax.set_title(f"Distribution SF moyenne - SNIR {snir_label}")
        ax.set_xlabel("Spreading Factor")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Nombre moyen de transmissions")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_learning_curve(learning_df: pd.DataFrame, output_path: Path) -> None:
    expected_columns = {"episode", "reward"}
    if not expected_columns.issubset(set(learning_df.columns)):
        raise ValueError(
            "learning_curve_ucb.csv doit contenir les colonnes episode,reward."
        )

    sorted_df = learning_df.sort_values("episode")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sorted_df["episode"], sorted_df["reward"], color="tab:purple", linewidth=2.0)
    ax.set_title("Learning curve UCB")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward normalisée")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root, figures_dir = _resolve_paths(args)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_required_csvs(output_root)

    _plot_metric_vs_n(
        frames["SNIR_OFF/pdr_results.csv"],
        frames["SNIR_ON/pdr_results.csv"],
        metric_col="pdr",
        ylabel="PDR",
        title="PDR vs N",
        output_path=figures_dir / f"pdr_vs_n.{args.format}",
    )
    _plot_metric_vs_n(
        frames["SNIR_OFF/throughput_results.csv"],
        frames["SNIR_ON/throughput_results.csv"],
        metric_col="throughput_packets_per_s",
        ylabel="Throughput (packets/s)",
        title="Throughput vs N",
        output_path=figures_dir / f"throughput_vs_n.{args.format}",
    )
    _plot_metric_vs_n(
        frames["SNIR_OFF/energy_results.csv"],
        frames["SNIR_ON/energy_results.csv"],
        metric_col="energy_joule_per_packet",
        ylabel="Énergie (J/packet)",
        title="Energy vs N",
        output_path=figures_dir / f"energy_vs_n.{args.format}",
    )

    _plot_sf_distribution(
        frames["SNIR_OFF/sf_distribution.csv"],
        frames["SNIR_ON/sf_distribution.csv"],
        output_path=figures_dir / f"sf_distribution.{args.format}",
    )

    _plot_learning_curve(
        frames["learning_curve_ucb.csv"],
        output_path=figures_dir / f"learning_curve_ucb.{args.format}",
    )

    _write_figures_manifest(
        manifest_path=figures_dir / "figures_manifest.json",
        campaign_id=args.campaign_id,
        output_root=output_root,
        figures_dir=figures_dir,
        format_name=args.format,
    )

    print(f"Figures générées dans: {figures_dir}")


if __name__ == "__main__":
    main()
