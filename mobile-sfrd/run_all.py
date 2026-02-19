"""Point d'entrée pour l'exécution complète des expérimentations mobile-sfrd."""

from __future__ import annotations

from pathlib import Path

from experiments import exp1_pdr_vs_speed, exp2_learning_curve, exp3_sf_hist, exp4_der_vs_speed, exp5_changepoint


def run(base_dir: str) -> dict[str, str]:
    """Exécute toutes les expériences et retourne les CSV produits."""
    return {
        "fig1": exp1_pdr_vs_speed.run(base_dir),
        "fig2": exp2_learning_curve.run(base_dir),
        "fig3": exp3_sf_hist.run(base_dir),
        "fig4": exp4_der_vs_speed.run(base_dir),
        "fig5": exp5_changepoint.run(base_dir),
    }


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)
    outputs = run(root)
    for name, csv_path in outputs.items():
        print(f"{name}: {csv_path}")
