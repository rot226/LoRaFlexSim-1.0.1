"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import csv
import importlib
import traceback
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

REQUIRED_ALGOS = {
    "step1": ("adr", "mixra_h", "mixra_opt"),
    "step2": ("adr", "mixra_h", "mixra_opt", "ucb1_sf"),
}

REQUIRED_SNIR_MODES = {
    "step1": ("snir_on", "snir_off"),
    "step2": ("snir_on",),
}

ALGO_ALIASES = {
    "adr": "adr",
    "mixra_h": "mixra_h",
    "mixra_hybrid": "mixra_h",
    "mixra_opt": "mixra_opt",
    "mixra_optimal": "mixra_opt",
    "mixraopt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "ucb1sf": "ucb1_sf",
}

SNIR_ALIASES = {
    "snir_on": "snir_on",
    "on": "snir_on",
    "true": "snir_on",
    "1": "snir_on",
    "yes": "snir_on",
    "snir_off": "snir_off",
    "off": "snir_off",
    "false": "snir_off",
    "0": "snir_off",
    "no": "snir_off",
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


def _run_plot_module(
    module_path: str,
    *,
    network_sizes: list[int] | None = None,
    allow_sample: bool = True,
) -> None:
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_path} sans fonction main().")
    if network_sizes is None:
        module.main(allow_sample=allow_sample)
        return
    module.main(network_sizes=network_sizes, allow_sample=allow_sample)


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _normalize_algo(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    if not cleaned:
        return None
    return ALGO_ALIASES.get(cleaned)


def _normalize_snir(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return SNIR_ALIASES.get(cleaned)


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


def _load_csv_data(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return ([], [])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    return (fieldnames, rows)


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


def _suggest_regeneration_command(path: Path, expected_sizes: list[int]) -> str | None:
    sizes = " ".join(str(size) for size in expected_sizes)
    if "step1" in path.parts:
        return (
            "python article_c/step1/run_step1.py "
            f"--network-sizes {sizes} --replications 5 --seeds_base 1000 "
            "--snir_modes snir_on,snir_off"
        )
    if "step2" in path.parts:
        return (
            "python article_c/step2/run_step2.py "
            f"--network-sizes {sizes} --replications 5 --seeds_base 1000"
        )
    return None


def _validate_network_sizes(paths: list[Path], expected_sizes: list[int]) -> bool:
    expected_set = {int(size) for size in expected_sizes}
    errors: list[str] = []
    for path in paths:
        found_sizes = _extract_network_sizes(path)
        missing = sorted(expected_set - found_sizes)
        if missing:
            missing_list = ", ".join(str(size) for size in missing)
            message_lines = [
                "ERREUR: tailles de réseau manquantes dans les résultats.",
                f"CSV: {path}",
                f"Tailles attendues manquantes: {missing_list}.",
            ]
            command = _suggest_regeneration_command(path, expected_sizes)
            if command:
                message_lines.append(
                    "Commande PowerShell pour régénérer les résultats:"
                )
                message_lines.append(command)
            errors.append("\n".join(message_lines))
    if errors:
        print(
            "\n\n".join(errors)
            + "\nAucun plot n'a été généré afin d'éviter des figures partielles."
        )
        return False
    return True


def _validate_plot_data(
    *,
    step: str,
    module_path: str,
    csv_path: Path,
    cached_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> bool:
    if not csv_path.exists():
        print(
            "AVERTISSEMENT: "
            f"CSV manquant pour {module_path}, figure ignorée."
        )
        return False
    if step not in cached_data:
        cached_data[step] = _load_csv_data(csv_path)
    fieldnames, rows = cached_data[step]
    if not fieldnames:
        print(
            "AVERTISSEMENT: "
            f"CSV vide pour {module_path}, figure ignorée."
        )
        return False
    sizes = _extract_network_sizes(csv_path)
    if len(sizes) < 2:
        print(
            "AVERTISSEMENT: "
            f"{module_path} nécessite au moins 2 tailles "
            "disponibles, figure ignorée."
        )
        print(f"CSV path: {csv_path}")
        return False
    algo_col = _pick_column(fieldnames, ("algo", "algorithm", "method"))
    snir_col = _pick_column(
        fieldnames, ("snir_mode", "snir_state", "snir", "with_snir")
    )
    if not algo_col or not snir_col:
        print(
            "AVERTISSEMENT: "
            f"{module_path} nécessite les colonnes algo/snir_mode, "
            "figure ignorée."
        )
        return False
    available_algos = {
        normalized
        for row in rows
        if (normalized := _normalize_algo(row.get(algo_col))) is not None
    }
    available_snir = {
        normalized
        for row in rows
        if (normalized := _normalize_snir(row.get(snir_col))) is not None
    }
    missing_algos = [
        algo for algo in REQUIRED_ALGOS[step] if algo not in available_algos
    ]
    missing_snir = [
        mode for mode in REQUIRED_SNIR_MODES[step] if mode not in available_snir
    ]
    if missing_algos or missing_snir:
        details = []
        if missing_algos:
            details.append(f"algos manquants: {', '.join(missing_algos)}")
        if missing_snir:
            details.append(f"SNIR manquants: {', '.join(missing_snir)}")
        print(
            "AVERTISSEMENT: "
            f"{module_path} incomplet ({' ; '.join(details)}), "
            "figure ignorée."
        )
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
    step_network_sizes: dict[str, list[int]] = {}
    if "step1" in steps:
        step_network_sizes["step1"] = _load_network_sizes_from_csvs([step1_csv])
    if "step2" in steps:
        step_network_sizes["step2"] = _load_network_sizes_from_csvs([step2_csv])
    if args.network_sizes:
        network_sizes = args.network_sizes
        if not _validate_network_sizes(csv_paths, network_sizes):
            return
    else:
        network_sizes = sorted(
            {
                size
                for sizes in step_network_sizes.values()
                for size in sizes
            }
        )
        if not network_sizes:
            print(
                "Aucune taille de réseau détectée dans les CSV, "
                "aucun plot n'a été généré."
            )
            return
    csv_cache: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for step in steps:
        for module_path in PLOT_MODULES[step]:
            csv_path = step1_csv if step == "step1" else step2_csv
            rl10_network_sizes: list[int] | None = None
            if (
                step == "step2"
                and module_path.endswith("plot_RL10_reward_vs_pdr_scatter")
            ):
                step1_sizes = _load_network_sizes_from_csvs([step1_csv])
                step2_sizes = _load_network_sizes_from_csvs([step2_csv])
                intersection = sorted(set(step1_sizes) & set(step2_sizes))
                if len(intersection) < 2:
                    print(
                        "ERREUR: "
                        "plot_RL10_reward_vs_pdr_scatter nécessite au moins "
                        "2 tailles communes entre Step1 et Step2."
                    )
                    print(f"Tailles Step1: {step1_sizes or 'aucune'}")
                    print(f"Tailles Step2: {step2_sizes or 'aucune'}")
                    print(f"Intersection: {intersection or 'aucune'}")
                    regen_sizes = step2_sizes or step1_sizes
                    command = (
                        _suggest_regeneration_command(
                            step1_csv,
                            regen_sizes,
                        )
                        if regen_sizes
                        else None
                    )
                    if command:
                        print(
                            "Exemple pour régénérer Step1 "
                            "(PowerShell):"
                        )
                        print(command)
                    continue
                rl10_network_sizes = intersection
            if not _validate_plot_data(
                step=step,
                module_path=module_path,
                csv_path=csv_path,
                cached_data=csv_cache,
            ):
                continue
            if step == "step2":
                figure = module_path.split(".")[-1]
                step2_sizes = (
                    rl10_network_sizes
                    or (
                        network_sizes
                        if args.network_sizes
                        else step_network_sizes.get("step2") or network_sizes
                    )
                )
                sizes_label = (
                    ", ".join(str(size) for size in step2_sizes)
                    if step2_sizes
                    else "none"
                )
                print(f"Detected sizes: {sizes_label}")
                print(f"Plotting Step2: {figure}")
            if step == "step2":
                step2_network_sizes = (
                    rl10_network_sizes
                    or (
                        network_sizes
                        if args.network_sizes
                        else step_network_sizes.get("step2") or network_sizes
                    )
                )
                try:
                    _run_plot_module(
                        module_path,
                        network_sizes=step2_network_sizes,
                        allow_sample=False,
                    )
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    traceback.print_exc()
            else:
                try:
                    _run_plot_module(module_path, allow_sample=False)
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    traceback.print_exc()


if __name__ == "__main__":
    main()
