"""Visualisation de la consommation énergétique en fonction du duty-cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.mne3sd.common import (
    apply_ieee_style,
    prepare_figure_directory,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "results" / "mne3sd" / "article_a" / "energy_consumption_summary.csv"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "mne3sd" / "article_a"
ARTICLE = "article_a"
SCENARIO = "energy_duty_cycle"
REQUIRED_COLUMNS = {
    "class",
    "duty_cycle",
    "energy_per_node_J_mean",
    "energy_per_node_J_std",
    "energy_per_message_J_mean",
    "energy_per_message_J_std",
    "energy_tx_per_node_J_mean",
    "energy_rx_per_node_J_mean",
    "energy_sleep_per_node_J_mean",
    "pdr_mean",
    "pdr_std",
}


def parse_arguments() -> argparse.Namespace:
    """Construire et analyser la ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Tracer les métriques énergétiques et de fiabilité en fonction du duty-cycle "
            "pour chaque classe LoRaWAN."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Chemin du fichier CSV energy_consumption_summary.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire de sortie pour les figures générées",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Activer le rendu LaTeX dans Matplotlib",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher les figures au lieu de les fermer en mode batch",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    """Lire le CSV d'agrégats énergétiques et valider ses colonnes."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de résultats introuvable : {path}. Exécutez le script de post-traitement au préalable."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["class"] = df["class"].astype(str)
    df["duty_cycle"] = pd.to_numeric(df["duty_cycle"], errors="coerce")

    numeric_columns = [
        "energy_per_node_J_mean",
        "energy_per_node_J_std",
        "energy_per_message_J_mean",
        "energy_per_message_J_std",
        "energy_tx_per_node_J_mean",
        "energy_rx_per_node_J_mean",
        "energy_sleep_per_node_J_mean",
        "pdr_mean",
        "pdr_std",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["duty_cycle"].isna().any():
        raise ValueError("Certaines lignes possèdent un duty-cycle invalide")

    return df.sort_values(["class", "duty_cycle"]).reset_index(drop=True)


def _resolve_figures_base(path: Path) -> Path:
    if path.name == ARTICLE:
        return path.parent
    return path


def plot_energy_per_node_vs_duty_cycle(df: pd.DataFrame, figures_base: Path) -> None:
    """Tracer l'énergie moyenne par nœud en fonction du duty-cycle."""

    fig, ax = plt.subplots()
    for class_name, group in df.groupby("class"):
        ordered = group.sort_values("duty_cycle")
        duty_cycle_pct = ordered["duty_cycle"] * 100.0
        ax.errorbar(
            duty_cycle_pct,
            ordered["energy_per_node_J_mean"],
            yerr=ordered["energy_per_node_J_std"],
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

    ax.set_xlabel("Duty cycle (%)")
    ax.set_ylabel("Energy per node (J)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Class")
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_per_node_vs_duty_cycle",
        base_dir=figures_base,
    )
    save_figure(fig, "energy_per_node_vs_duty_cycle", output_dir)


def plot_pdr_vs_duty_cycle(df: pd.DataFrame, figures_base: Path) -> None:
    """Tracer le PDR moyen en fonction du duty-cycle."""

    fig, ax = plt.subplots()
    for class_name, group in df.groupby("class"):
        ordered = group.sort_values("duty_cycle")
        duty_cycle_pct = ordered["duty_cycle"] * 100.0
        pdr_pct = ordered["pdr_mean"] * 100.0
        pdr_std_pct = ordered["pdr_std"] * 100.0
        ax.errorbar(
            duty_cycle_pct,
            pdr_pct,
            yerr=pdr_std_pct,
            marker="o",
            capsize=3,
            label=f"Class {class_name}",
        )

    ax.set_xlabel("Duty cycle (%)")
    ax.set_ylabel("Packet delivery ratio (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Class")
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="pdr_vs_duty_cycle",
        base_dir=figures_base,
    )
    save_figure(fig, "pdr_vs_duty_cycle", output_dir)


def plot_energy_breakdown(df: pd.DataFrame, figures_base: Path) -> None:
    """Tracer la décomposition énergétique (TX/RX/Sommeil) par classe et duty-cycle."""

    ordered = df.sort_values(["class", "duty_cycle"])  # garantir l'ordre sur l'axe X
    labels = [
        f"Class {row['class']}\n{row['duty_cycle'] * 100:.1f}%" for _, row in ordered.iterrows()
    ]
    x = range(len(ordered))

    fig, ax = plt.subplots()
    width = 0.8

    tx = ordered["energy_tx_per_node_J_mean"]
    rx = ordered["energy_rx_per_node_J_mean"]
    sleep = ordered["energy_sleep_per_node_J_mean"]

    ax.bar(x, tx, width=width, label="TX")
    ax.bar(x, rx, width=width, bottom=tx, label="RX")
    ax.bar(x, sleep, width=width, bottom=tx + rx, label="Sleep")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Energy per node (J)")
    ax.set_xlabel("LoRaWAN class and duty cycle")
    ax.legend(title="Component")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    output_dir = prepare_figure_directory(
        article=ARTICLE,
        scenario=SCENARIO,
        metric="energy_breakdown_vs_duty_cycle",
        base_dir=figures_base,
    )
    save_figure(fig, "energy_breakdown_vs_duty_cycle", output_dir)


def main() -> None:
    args = parse_arguments()

    apply_ieee_style()
    if args.latex:
        plt.rcParams.update({"text.usetex": True})

    summary = load_summary(args.input)

    figures_base = _resolve_figures_base(args.figures_dir)

    plot_energy_per_node_vs_duty_cycle(summary, figures_base)
    plot_pdr_vs_duty_cycle(summary, figures_base)
    plot_energy_breakdown(summary, figures_base)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover - point d'entrée du script
    main()
