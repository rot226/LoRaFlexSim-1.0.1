"""Orchestre les comparaisons de graphiques pour l'article C.

Ce script enchaîne :
- reproduce_author_results.py
- compare_with_snir.py
- plot_cluster_der.py

Il centralise les formats d'export et les répertoires de sortie, puis affiche
un résumé des fichiers produits pour faciliter le post-traitement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STEP1 = Path("article_c/step1/results/aggregated_results.csv")
DEFAULT_STEP2 = Path("article_c/step2/results/aggregated_results.csv")


def _split_formats(value: str) -> list[str]:
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _run_command(cmd: list[str], label: str) -> None:
    print(f"\n==> {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _collect_outputs(output_dir: Path, formats: list[str]) -> list[str]:
    if not output_dir.exists():
        return []
    files: list[str] = []
    for fmt in formats:
        files.extend(
            str(path.relative_to(output_dir))
            for path in sorted(output_dir.rglob(f"*.{fmt}"))
        )
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lance les scripts de comparaison des plots article C."
    )
    parser.add_argument(
        "--step1-results",
        type=Path,
        default=DEFAULT_STEP1,
        help="Chemin vers aggregated_results.csv (step1).",
    )
    parser.add_argument(
        "--step2-results",
        type=Path,
        default=DEFAULT_STEP2,
        help="Chemin vers aggregated_results.csv (step2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("article_c/plots/output/compare_all"),
        help="Répertoire racine pour les sorties générées.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Formats d'export (ex: png,pdf,eps).",
    )
    parser.add_argument(
        "--snir-modes",
        type=str,
        default="snir_on,snir_off",
        help="Modes SNIR pour compare_with_snir (ex: snir_on,snir_off).",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=None,
        help="Filtre sur un seuil SNIR précis (dB).",
    )
    parser.add_argument(
        "--snir-threshold-min-db",
        type=float,
        default=None,
        help="Borne basse de clamp SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-max-db",
        type=float,
        default=None,
        help="Borne haute de clamp SNIR (dB).",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="all",
        help="Cluster à filtrer pour compare_with_snir (défaut: all).",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        help="Filtrer les clusters pour plot_cluster_der (ex: gold silver).",
    )
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau pour plot_cluster_der.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    formats = _split_formats(args.formats)
    if not formats:
        raise ValueError("Aucun format d'export fourni.")

    output_root = args.output_dir
    output_author = output_root / "reproduce_author_results"
    output_snir = output_root / "compare_with_snir"
    output_cluster = output_root / "plot_cluster_der"

    output_root.mkdir(parents=True, exist_ok=True)
    output_author.mkdir(parents=True, exist_ok=True)
    output_snir.mkdir(parents=True, exist_ok=True)
    output_cluster.mkdir(parents=True, exist_ok=True)

    reproduce_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "reproduce_author_results.py"),
        "--step1-results",
        str(args.step1_results),
        "--step2-results",
        str(args.step2_results),
        "--output-dir",
        str(output_author),
        "--formats",
        ",".join(formats),
    ]
    compare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compare_with_snir.py"),
        "--step1-csv",
        str(args.step1_results),
        "--step2-csv",
        str(args.step2_results),
        "--output-dir",
        str(output_snir),
        "--formats",
        ",".join(formats),
        "--snir-modes",
        args.snir_modes,
        "--cluster",
        args.cluster,
    ]
    plot_cluster_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_cluster_der.py"),
        "--output-dir",
        str(output_cluster),
        "--formats",
        ",".join(formats),
    ]

    if args.snir_threshold_db is not None:
        reproduce_cmd += ["--snir-threshold-db", str(args.snir_threshold_db)]
        compare_cmd += ["--snir-threshold-db", str(args.snir_threshold_db)]
    if args.snir_threshold_min_db is not None:
        reproduce_cmd += ["--snir-threshold-min-db", str(args.snir_threshold_min_db)]
        compare_cmd += ["--snir-threshold-min-db", str(args.snir_threshold_min_db)]
    if args.snir_threshold_max_db is not None:
        reproduce_cmd += ["--snir-threshold-max-db", str(args.snir_threshold_max_db)]
        compare_cmd += ["--snir-threshold-max-db", str(args.snir_threshold_max_db)]

    if args.clusters:
        plot_cluster_cmd += ["--clusters", *args.clusters]
    if args.network_sizes:
        plot_cluster_cmd += ["--network-sizes", *map(str, args.network_sizes)]

    _run_command(reproduce_cmd, "Reproduction des résultats auteurs")
    _run_command(compare_cmd, "Comparaison SNIR")
    _run_command(plot_cluster_cmd, "DER par cluster")

    summary = {
        "output_root": str(output_root),
        "formats": formats,
        "files": {
            "reproduce_author_results": _collect_outputs(output_author, formats),
            "compare_with_snir": _collect_outputs(output_snir, formats),
            "plot_cluster_der": _collect_outputs(output_cluster, formats),
        },
    }
    summary["all_files"] = sorted(
        {
            *summary["files"]["reproduce_author_results"],
            *summary["files"]["compare_with_snir"],
            *summary["files"]["plot_cluster_der"],
        }
    )

    print("\nRésumé des fichiers générés (relatif à chaque sous-dossier):")
    for section, files in summary["files"].items():
        print(f"- {section}:")
        if not files:
            print("  (aucun fichier)")
            continue
        for filename in files:
            print(f"  - {filename}")

    print("\nRésumé JSON:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
