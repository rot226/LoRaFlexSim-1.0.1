"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import pandas as pd


PLOT_MODULES = {
    "step1": [
        "article_c.step1.plots.plot_S1",
        "article_c.step1.plots.plot_S2",
        "article_c.step1.plots.plot_S3",
        "article_c.step1.plots.plot_S4",
        "article_c.step1.plots.plot_S5",
        "article_c.step1.plots.plot_S6",
        "article_c.step1.plots.plot_S6_cluster_pdr_vs_density",
        "article_c.step1.plots.plot_S6_cluster_pdr_vs_network_size",
        "article_c.step1.plots.plot_S7_cluster_outage_vs_density",
        "article_c.step1.plots.plot_S7_cluster_outage_vs_network_size",
        "article_c.step1.plots.plot_S8_spreading_factor_distribution",
        "article_c.step1.plots.plot_S9_latency_or_toa_vs_network_size",
    ],
    "step2": [
        "article_c.step2.plots.plot_RL1",
        "article_c.step2.plots.plot_RL1_learning_curve_reward",
        "article_c.step2.plots.plot_RL2",
        "article_c.step2.plots.plot_RL3",
        "article_c.step2.plots.plot_RL4",
        "article_c.step2.plots.plot_RL5",
        "article_c.step2.plots.plot_RL6_cluster_outage_vs_density",
        "article_c.step2.plots.plot_RL7_reward_vs_density",
        "article_c.step2.plots.plot_RL8_reward_distribution",
        "article_c.step2.plots.plot_RL9_sf_selection_entropy",
        "article_c.step2.plots.plot_RL10_reward_vs_pdr_scatter",
    ],
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour générer les figures."""
    parser = argparse.ArgumentParser(
        description="Génère toutes les figures à partir des CSV agrégés."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="step1,step2",
        help="Étapes à tracer (ex: step1,step2).",
    )
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Tailles de réseau attendues (ex: 50 100 150).",
    )
    return parser


def _parse_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [step for step in steps if step not in PLOT_MODULES]
    if unknown:
        raise ValueError(f"Étape(s) inconnue(s): {', '.join(unknown)}")
    return steps


def _run_plot_module(module_path: str) -> None:
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_path} sans fonction main().")
    module.main()


def _validate_snir_mode_column(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
        if "snir_mode" not in fieldnames:
            raise ValueError(
                f"Le CSV {path} doit contenir une colonne 'snir_mode'."
            )


def _extract_network_sizes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "network_size" in fieldnames:
            size_key = "network_size"
        elif "density" in fieldnames:
            size_key = "density"
        else:
            return set()
        sizes: set[int] = set()
        for row in reader:
            raw_value = row.get(size_key)
            if raw_value in (None, ""):
                continue
            try:
                sizes.add(int(float(raw_value)))
            except ValueError:
                continue
    return sizes


def _load_network_sizes_from_csvs(paths: list[Path]) -> list[int]:
    sizes: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "network_size" not in df.columns:
            raise ValueError(
                f"Le CSV {path} doit contenir une colonne 'network_size'."
            )
        sizes.update(
            int(value) for value in df["network_size"].dropna().unique().tolist()
        )
    return sorted(sizes)


def _validate_network_sizes(paths: list[Path], expected_sizes: list[int]) -> bool:
    expected_set = {int(size) for size in expected_sizes}
    for path in paths:
        found_sizes = _extract_network_sizes(path)
        missing = sorted(expected_set - found_sizes)
        if missing:
            missing_list = ", ".join(str(size) for size in missing)
            print(f"Missing sizes: {missing_list}")
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    article_dir = Path(__file__).resolve().parent
    step1_results_dir = article_dir / "step1" / "results"
    step2_results_dir = article_dir / "step2" / "results"
    if "step1" in steps and not (step1_results_dir / "done.flag").exists():
        print("Step1 incomplete, skipping plots")
        return
    if "step2" in steps and not (step2_results_dir / "done.flag").exists():
        print("Step2 incomplete, skipping plots")
        return
    step1_csv = article_dir / "step1" / "results" / "aggregated_results.csv"
    step2_csv = article_dir / "step2" / "results" / "aggregated_results.csv"
    csv_paths: list[Path] = []
    if "step1" in steps:
        csv_paths.append(step1_csv)
    if "step2" in steps:
        if step2_csv.exists():
            csv_paths.append(step2_csv)
        else:
            print(
                "CSV Step2 absent : "
                f"{step2_csv} introuvable. "
                "Exécutez l'étape 2 pour générer aggregated_results.csv "
                "avant de lancer les plots Step2."
            )
            steps = [step for step in steps if step != "step2"]
    _validate_snir_mode_column(csv_paths)
    if args.network_sizes:
        network_sizes = args.network_sizes
        if not _validate_network_sizes(csv_paths, network_sizes):
            return
    else:
        network_sizes = _load_network_sizes_from_csvs(csv_paths)
    for step in steps:
        for module_path in PLOT_MODULES[step]:
            if step == "step2":
                figure = module_path.split(".")[-1]
                print(f"Plotting Step2: {figure}")
            _run_plot_module(module_path)


if __name__ == "__main__":
    main()
