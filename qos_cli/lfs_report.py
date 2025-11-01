"""CLI pour générer un rapport Markdown à partir des sorties QoS de LoRaFlexSim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class ReportInputs:
    """Regroupe les entrées nécessaires à la génération du rapport."""

    results_root: Path
    summary_file: Path
    figures_dir: Path
    report_file: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> ReportInputs:
    """Analyse les arguments de la ligne de commande."""

    parser = argparse.ArgumentParser(
        prog="lfs_report",
        description=(
            "Assemble un rapport Markdown en combinant la synthèse textuelle et les "
            "figures générées par lfs_plots.py."
        ),
    )
    parser.add_argument(
        "--in",
        dest="results_root",
        type=Path,
        required=True,
        help="Dossier racine contenant les résultats agrégés (<méthode>/<scénario>).",
    )
    parser.add_argument(
        "--summary",
        dest="summary_file",
        type=Path,
        default=Path("qos_cli") / "SUMMARY.txt",
        help="Fichier texte de synthèse (défaut : qos_cli/SUMMARY.txt).",
    )
    parser.add_argument(
        "--figures",
        dest="figures_dir",
        type=Path,
        default=Path("qos_cli") / "figures",
        help="Dossier contenant les figures PNG générées (défaut : qos_cli/figures).",
    )
    parser.add_argument(
        "--out",
        dest="report_file",
        type=Path,
        default=Path("qos_cli") / "REPORT.md",
        help="Chemin du rapport Markdown à produire (défaut : qos_cli/REPORT.md).",
    )

    args = parser.parse_args(argv)
    return ReportInputs(
        results_root=args.results_root,
        summary_file=args.summary_file,
        figures_dir=args.figures_dir,
        report_file=args.report_file,
    )


def validate_inputs(inputs: ReportInputs) -> None:
    """Vérifie l'existence des chemins indispensables."""

    if not inputs.results_root.exists():
        raise FileNotFoundError(
            f"Le dossier de résultats '{inputs.results_root}' est introuvable."
        )
    if not inputs.summary_file.exists():
        raise FileNotFoundError(
            f"Le fichier de synthèse '{inputs.summary_file}' est introuvable."
        )
    if not inputs.figures_dir.exists():
        # Les figures sont optionnelles, mais avertir si le dossier manque.
        inputs.figures_dir.mkdir(parents=True, exist_ok=True)


def load_summary(summary_file: Path) -> str:
    """Charge le contenu textuel du fichier SUMMARY."""

    return summary_file.read_text(encoding="utf-8").strip()


def collect_scenarios(results_root: Path) -> List[str]:
    """Identifie les scénarios présents sous le dossier de résultats."""

    scenarios: set[str] = set()
    if results_root.is_dir():
        for method_dir in results_root.iterdir():
            if not method_dir.is_dir():
                continue
            for scenario_dir in method_dir.iterdir():
                if scenario_dir.is_dir():
                    scenarios.add(scenario_dir.name)
    return sorted(scenarios)


def discover_figures(figures_dir: Path) -> List[Path]:
    """Récupère les fichiers PNG disponibles pour intégration au rapport."""

    if not figures_dir.exists():
        return []
    return sorted(path for path in figures_dir.glob("*.png") if path.is_file())


def make_relative_paths(paths: Iterable[Path], reference: Path) -> List[Path]:
    """Convertit une collection de chemins en chemins relatifs au rapport."""

    relative_paths: List[Path] = []
    for path in paths:
        try:
            relative_paths.append(path.relative_to(reference))
        except ValueError:
            try:
                relative_paths.append(path.resolve().relative_to(reference.resolve()))
            except ValueError:
                relative_paths.append(path)
    return relative_paths


def build_report(inputs: ReportInputs) -> str:
    """Assemble le contenu Markdown complet du rapport."""

    summary_content = load_summary(inputs.summary_file)
    scenarios = collect_scenarios(inputs.results_root)
    figures = discover_figures(inputs.figures_dir)
    report_parent = inputs.report_file.parent
    report_parent.mkdir(parents=True, exist_ok=True)
    relative_figures = make_relative_paths(figures, report_parent)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenarios_line = ", ".join(scenarios) if scenarios else "Aucun scénario détecté"

    sections = [
        "# Rapport QoS LoRaFlexSim",
        "",
        "## Métadonnées",
        f"- Date de génération : {timestamp}",
        f"- Dossier des résultats : {inputs.results_root}",
        f"- Scénarios analysés : {scenarios_line}",
        "",
        "## Synthèse",
        summary_content or "*(Synthèse vide)*",
        "",
        "## Figures",
    ]

    if relative_figures:
        for figure in relative_figures:
            title = figure.stem.replace("_", " ")
            sections.append(f"![{title}]({figure.as_posix()})")
    else:
        sections.append("Aucune figure PNG n'a été trouvée dans le dossier fourni.")

    sections.extend(
        [
            "",
            "## TODO",
            "- [ ] Automatiser la conversion du rapport Markdown vers PDF (ex. via Pandoc).",
        ]
    )

    return "\n".join(sections) + "\n"


def write_report(content: str, report_file: Path) -> None:
    """Écrit le contenu Markdown dans le fichier cible."""

    report_file.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Point d'entrée principal de la CLI lfs_report."""

    inputs = parse_args(argv)
    validate_inputs(inputs)
    report_content = build_report(inputs)
    write_report(report_content, inputs.report_file)


if __name__ == "__main__":
    main()
