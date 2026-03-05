"""CLI de génération des figures de campagne SFRD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_METRIC_SPECS = {
    "pdr_results.csv": {
        "figure_name": "pdr_vs_n",
        "metric_col": "pdr",
        "ylabel": "PDR",
        "title": "PDR vs N",
    },
    "throughput_results.csv": {
        "figure_name": "throughput_vs_n",
        "metric_col": "throughput_packets_per_s",
        "ylabel": "Throughput (packets/s)",
        "title": "Throughput vs N",
    },
    "energy_results.csv": {
        "figure_name": "energy_vs_n",
        "metric_col": "energy_joule_per_packet",
        "ylabel": "Énergie (J/packet)",
        "title": "Energy vs N",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Génère automatiquement les figures depuis SNIR_OFF/*.csv, "
            "SNIR_ON/*.csv et learning_curve_ucb.csv."
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
        help="Dossier de sortie personnalisé (sinon: figures/<campaign_id>/).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Format des figures (défaut: png).",
    )
    return parser


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    if not args.campaign_id:
        raise ValueError("Fournir --campaign-id ou --output-root.")
    return (args.logs_root / args.campaign_id / "output").resolve()


def _resolve_campaign_id(args: argparse.Namespace, output_root: Path) -> str:
    if args.campaign_id:
        return args.campaign_id
    inferred_campaign_id = output_root.parent.name
    if inferred_campaign_id:
        return inferred_campaign_id
    raise ValueError(
        "Impossible d'inférer campaign_id depuis --output-root. Fournir --campaign-id explicitement."
    )


def _resolve_figures_dir(args: argparse.Namespace, campaign_id: str) -> Path:
    if args.figures_dir is not None:
        return args.figures_dir.resolve()
    return (Path("figures") / campaign_id).resolve()


def _discover_csv_files(output_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    required_dirs = ("SNIR_OFF", "SNIR_ON")
    missing_dirs = [name for name in required_dirs if not (output_root / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Dossiers requis manquants pour le plotting:\n- "
            + "\n- ".join(str(output_root / missing_dir) for missing_dir in missing_dirs)
        )

    for snir_dir in required_dirs:
        for csv_path in sorted((output_root / snir_dir).glob("*.csv")):
            discovered[f"{snir_dir}/{csv_path.name}"] = csv_path

    learning_curve_path = output_root / "learning_curve_ucb.csv"
    if learning_curve_path.is_file():
        discovered["learning_curve_ucb.csv"] = learning_curve_path

    if not discovered:
        raise FileNotFoundError(
            f"Aucun CSV détecté dans {output_root} (attendu: SNIR_OFF/*.csv, SNIR_ON/*.csv, learning_curve_ucb.csv)."
        )
    return discovered


def _build_figure_entries(discovered_csvs: dict[str, Path], format_name: str) -> list[dict[str, str | list[str]]]:
    entries: list[dict[str, str | list[str]]] = []

    for file_name, metadata in _METRIC_SPECS.items():
        off_rel = f"SNIR_OFF/{file_name}"
        on_rel = f"SNIR_ON/{file_name}"
        if off_rel in discovered_csvs and on_rel in discovered_csvs:
            entries.append(
                {
                    "name": str(metadata["figure_name"]),
                    "file_name": f"{metadata['figure_name']}.{format_name}",
                    "source_csv_rel": [off_rel, on_rel],
                }
            )

    sf_off = "SNIR_OFF/sf_distribution.csv"
    sf_on = "SNIR_ON/sf_distribution.csv"
    if sf_off in discovered_csvs and sf_on in discovered_csvs:
        entries.append(
            {
                "name": "sf_distribution",
                "file_name": f"sf_distribution.{format_name}",
                "source_csv_rel": [sf_off, sf_on],
            }
        )

    if "learning_curve_ucb.csv" in discovered_csvs:
        entries.append(
            {
                "name": "learning_curve_ucb",
                "file_name": f"learning_curve_ucb.{format_name}",
                "source_csv_rel": ["learning_curve_ucb.csv"],
            }
        )
    return entries


def _load_csv_frames(discovered_csvs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        relative_path: pd.read_csv(csv_path)
        for relative_path, csv_path in discovered_csvs.items()
    }


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


def _plot_sf_distribution(off_df: pd.DataFrame, on_df: pd.DataFrame, output_path: Path) -> None:
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
    if not {"episode", "reward"}.issubset(set(learning_df.columns)):
        raise ValueError("learning_curve_ucb.csv doit contenir les colonnes episode,reward.")

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


def _write_figures_manifest(
    *,
    manifest_path: Path,
    campaign_id: str,
    output_root: Path,
    figures_dir: Path,
    format_name: str,
    discovered_csvs: dict[str, Path],
    figure_entries: list[dict[str, str | list[str]]],
) -> None:
    figures = []
    for entry in figure_entries:
        source_csv_rel = [str(rel) for rel in entry["source_csv_rel"]]
        source_csv_abs = [str((output_root / rel).resolve()) for rel in source_csv_rel]
        figures.append(
            {
                "name": str(entry["name"]),
                "file": str((figures_dir / str(entry["file_name"])).resolve()),
                "source_csv_rel": source_csv_rel,
                "source_csv": source_csv_abs,
            }
        )

    payload = {
        "campaign_id": campaign_id,
        "output_root": str(output_root.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "format": format_name,
        "discovered_csv": sorted(discovered_csvs.keys()),
        "figures": figures,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = _resolve_output_root(args)
    campaign_id = _resolve_campaign_id(args, output_root)
    figures_dir = _resolve_figures_dir(args, campaign_id)
    figures_dir.mkdir(parents=True, exist_ok=True)

    discovered_csvs = _discover_csv_files(output_root)
    frames = _load_csv_frames(discovered_csvs)
    figure_entries = _build_figure_entries(discovered_csvs, args.format)
    if not figure_entries:
        raise FileNotFoundError(
            "Aucune figure générable: vérifier la présence de couples SNIR_OFF/SNIR_ON pour les métriques et/ou learning_curve_ucb.csv."
        )

    metric_by_figure_name = {
        str(meta["figure_name"]): meta for meta in _METRIC_SPECS.values()
    }

    for entry in figure_entries:
        name = str(entry["name"])
        output_path = figures_dir / str(entry["file_name"])
        source_csv_rel = [str(rel) for rel in entry["source_csv_rel"]]

        if name in metric_by_figure_name:
            metric_meta = metric_by_figure_name[name]
            _plot_metric_vs_n(
                frames[source_csv_rel[0]],
                frames[source_csv_rel[1]],
                metric_col=str(metric_meta["metric_col"]),
                ylabel=str(metric_meta["ylabel"]),
                title=str(metric_meta["title"]),
                output_path=output_path,
            )
        elif name == "sf_distribution":
            _plot_sf_distribution(
                frames[source_csv_rel[0]],
                frames[source_csv_rel[1]],
                output_path=output_path,
            )
        elif name == "learning_curve_ucb":
            _plot_learning_curve(frames[source_csv_rel[0]], output_path=output_path)

    _write_figures_manifest(
        manifest_path=figures_dir / "figures_manifest.json",
        campaign_id=campaign_id,
        output_root=output_root,
        figures_dir=figures_dir,
        format_name=args.format,
        discovered_csvs=discovered_csvs,
        figure_entries=figure_entries,
    )

    print(f"Figures générées dans: {figures_dir}")


if __name__ == "__main__":
    main()
