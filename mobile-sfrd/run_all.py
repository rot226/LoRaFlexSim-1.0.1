"""Point d'entrée pour l'exécution complète des expérimentations mobile-sfrd."""

from __future__ import annotations

from pathlib import Path

from core.utils import ensure_output_dirs
from experiments import (
    exp1_pdr_vs_speed,
    exp2_learning_curve,
    exp3_sf_hist,
    exp4_der_vs_speed,
    exp5_changepoint,
)


def _list_generated_files(directory: Path) -> list[Path]:
    """Retourne les fichiers présents dans un dossier, triés par nom."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file())


def run(base_dir: str | None = None) -> dict[str, str]:
    """Exécute toutes les expériences dans l'ordre demandé."""
    project_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
    dirs = ensure_output_dirs(str(project_dir))

    outputs = {
        "exp1": exp1_pdr_vs_speed.run(str(project_dir)),
        "exp2": exp2_learning_curve.run(str(project_dir)),
        "exp3": exp3_sf_hist.run(str(project_dir)),
        "exp4": exp4_der_vs_speed.run(str(project_dir)),
        "exp5": exp5_changepoint.run(str(project_dir)),
    }

    csv_files = _list_generated_files(Path(dirs["csv"]))
    figure_files = _list_generated_files(Path(dirs["figures"]))

    print("\nRésumé final des fichiers générés")
    print(f"- outputs/csv ({len(csv_files)} fichier(s))")
    for path in csv_files:
        print(f"  - {path.name}")

    print(f"- outputs/figures ({len(figure_files)} fichier(s))")
    for path in figure_files:
        print(f"  - {path.name}")

    return outputs


if __name__ == "__main__":
    run()
