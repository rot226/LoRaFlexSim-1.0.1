"""Visualisations des métriques énergétiques et de fiabilité en downlink."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.mne3sd.common import apply_ieee_style, prepare_figure_directory, save_figure

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "results" / "mne3sd" / "article_a" / "class_downlink_energy.csv"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "mne3sd" / "article_a"
ARTICLE = "article_a"
SCENARIO = "class_downlink_energy"
REQUIRED_COLUMNS = {
    "class",
    "replicate",
    "uplink_pdr",
    "downlink_pdr",
    "energy_tx_J",
    "energy_rx_J",
    "energy_idle_J",
}


def parse_arguments() -> argparse.Namespace:
    """Construire et analyser la ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Tracer la décomposition énergétique TX/RX/veille et comparer les taux de "
            "réception montants/descendants pour chaque classe LoRaWAN."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Chemin du fichier CSV class_downlink_energy.csv produit par le scénario.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire racine dans lequel enregistrer les figures générées.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher les figures au terme de l'exécution (mode interactif).",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    """Charger le CSV et retourner uniquement les agrégats par classe."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de résultats introuvable : {path}. Exécutez d'abord le scénario associé."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["class"] = df["class"].astype(str)
    df["replicate"] = df["replicate"].astype(str)

    numeric_columns = ["uplink_pdr", "downlink_pdr", "energy_tx_J", "energy_rx_J", "energy_idle_J"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary_mask = df["replicate"].str.lower() == "mean"
    if summary_mask.any():
        summary = df.loc[summary_mask].copy()
    else:
        summary = df.groupby("class", as_index=False)[numeric_columns].mean()
        summary["replicate"] = "mean"

    summary = summary.dropna(subset=["class"]).sort_values("class").reset_index(drop=True)
    return summary


def _resolve_figures_base(path: Path) -> Path:
    if path.name == ARTICLE:
        return path.parent
    return path


def plot_energy_breakdown(
    summary: pd.DataFrame, figures_base: Path, *, close: bool = True
) -> plt.Figure:
    """Tracer un histogramme empilé TX/RX/veille pour chaque classe."""

    classes = summary["class"].tolist()
    indices = np.arange(len(classes))

    tx = summary["energy_tx_J"].to_numpy()
    rx = summary["energy_rx_J"].to_numpy()
    idle = summary["energy_idle_J"].to_numpy()

    fig, ax = plt.subplots()
    width = 0.6

    ax.bar(indices, tx, width=width, label="TX")
    ax.bar(indices, rx, width=width, bottom=tx, label="RX")
    ax.bar(indices, idle, width=width, bottom=tx + rx, label="Idle")

    ax.set_xticks(indices)
    ax.set_xticklabels([f"Class {name}" for name in classes])
    ax.set_xlabel("LoRaWAN class")
    ax.set_ylabel("Average energy per node (J)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_breakdown",
        base_dir=figures_base,
    )
    save_figure(fig, "energy_breakdown", output_dir)
    if close:
        plt.close(fig)
    return fig


def plot_pdr_comparison(
    summary: pd.DataFrame, figures_base: Path, *, close: bool = True
) -> plt.Figure:
    """Tracer un histogramme comparant le PDR montant et descendant."""

    classes = summary["class"].tolist()
    indices = np.arange(len(classes))
    width = 0.35

    uplink = (summary["uplink_pdr"].to_numpy() * 100.0).clip(min=0.0)
    downlink = (summary["downlink_pdr"].to_numpy() * 100.0).clip(min=0.0)

    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    ax.bar(indices - width / 2, uplink, width=width, label="Uplink PDR (%)")
    ax.bar(indices + width / 2, downlink, width=width, label="Downlink PDR (%)")

    ax.set_xticks(indices)
    ax.set_xticklabels([f"Class {name}" for name in classes])
    ax.set_xlabel("LoRaWAN class")
    ax.set_ylabel("Packet Delivery ratio (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="pdr_comparison",
        base_dir=figures_base,
    )
    save_figure(fig, "pdr_comparison", output_dir)
    if close:
        plt.close(fig)
    return fig


def main() -> None:
    args = parse_arguments()
    apply_ieee_style()

    summary = load_summary(args.input)
    figures_base = _resolve_figures_base(args.figures_dir)

    figures: list[plt.Figure] = []
    figures.append(
        plot_energy_breakdown(summary, figures_base, close=not args.show)
    )
    figures.append(
        plot_pdr_comparison(summary, figures_base, close=not args.show)
    )

    if args.show:
        plt.show()
    else:
        for fig in figures:
            if plt.fignum_exists(fig.number):
                plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    main()
