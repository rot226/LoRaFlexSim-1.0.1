"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import traceback
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    if find_spec("article_c") is None:
        raise ModuleNotFoundError(
            "Impossible d'importer 'article_c'. "
            "Ajoutez la racine du dépôt au PYTHONPATH."
        )

from article_c.common.csv_io import write_simulation_results, write_step1_results

ARTICLE_DIR = Path(__file__).resolve().parent
STEP1_RESULTS_DIR = ARTICLE_DIR / "step1" / "results"
STEP2_RESULTS_DIR = ARTICLE_DIR / "step2" / "results"

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


def _coerce_csv_value(value: str | None) -> object:
    if value is None or value == "":
        return ""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _load_csv_rows_coerced(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def _ensure_expected_results_dir(csv_path: Path, expected_dir: Path, label: str) -> None:
    try:
        csv_path.resolve().relative_to(expected_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "Chemin CSV inattendu pour "
            f"{label}: {csv_path} (attendu sous {expected_dir})."
        ) from exc


def _collect_nested_csvs(results_dir: Path, filename: str) -> list[Path]:
    return sorted(results_dir.glob(f"size_*/rep_*/{filename}"))


def _ensure_step1_aggregated(results_dir: Path) -> Path | None:
    aggregated_path = results_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        return aggregated_path
    raw_paths = _collect_nested_csvs(results_dir, "raw_metrics.csv")
    if not raw_paths:
        return None
    raw_rows: list[dict[str, object]] = []
    for path in raw_paths:
        raw_rows.extend(_load_csv_rows_coerced(path))
    if not raw_rows:
        return None
    print("Assemblage des résultats Step1 à partir des sous-dossiers...")
    write_step1_results(results_dir, raw_rows)
    return aggregated_path


def _ensure_step2_aggregated(results_dir: Path) -> Path | None:
    aggregated_path = results_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        return aggregated_path
    raw_paths = _collect_nested_csvs(results_dir, "raw_results.csv")
    if not raw_paths:
        return None
    raw_rows: list[dict[str, object]] = []
    for path in raw_paths:
        raw_rows.extend(_load_csv_rows_coerced(path))
    if not raw_rows:
        return None
    print("Assemblage des résultats Step2 à partir des sous-dossiers...")
    write_simulation_results(results_dir, raw_rows)
    return aggregated_path


def _report_missing_csv(step_label: str, results_dir: Path) -> None:
    aggregated_path = results_dir / "aggregated_results.csv"
    raw_metrics_path = results_dir / "raw_metrics.csv"
    raw_results_path = results_dir / "raw_results.csv"
    print(
        f"ERREUR: CSV {step_label} introuvable. "
        f"Chemin attendu: {aggregated_path}."
    )
    print(
        "INFO: vérifiez que les simulations ont bien écrit dans "
        f"{results_dir}."
    )
    if step_label == "Step1":
        print(f"INFO: exemples attendus: {raw_metrics_path}")
    else:
        print(f"INFO: exemples attendus: {raw_results_path}")


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
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Formats d'export des figures (ex: png,pdf,eps).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=(
            "Retourne un code non nul si des plots échouent, "
            "sans interrompre l'exécution."
        ),
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


def _validate_plot_modules_use_save_figure() -> dict[str, str]:
    missing: dict[str, str] = {}
    for module_paths in PLOT_MODULES.values():
        for module_path in module_paths:
            spec = find_spec(module_path)
            if spec is None or spec.origin is None:
                missing[module_path] = "module introuvable"
                continue
            source_path = Path(spec.origin)
            try:
                source = source_path.read_text(encoding="utf-8")
            except OSError as exc:
                missing[module_path] = f"lecture impossible: {exc}"
                continue
            if "save_figure(" not in source:
                missing[module_path] = "ne passe pas par save_figure"
    if missing:
        print(
            "ERREUR: certains scripts de plot ne passent pas par save_figure:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    return missing


def _inspect_plot_outputs(
    output_dir: Path,
    label: str,
    formats: list[str],
) -> None:
    if not output_dir.exists():
        print(
            "AVERTISSEMENT: "
            f"dossier de sortie absent pour {label}: {output_dir}"
        )
        return
    if not formats:
        print(f"INFO: aucun format d'export fourni pour {label}.")
        return
    primary_format = formats[0]
    primary_files = sorted(output_dir.glob(f"*.{primary_format}"))
    if not primary_files:
        print(
            "AVERTISSEMENT: "
            f"aucun {primary_format.upper()} trouvé pour {label} dans {output_dir}."
        )
        return
    missing_variants: list[str] = []
    for primary_path in primary_files:
        stem = primary_path.stem
        for ext in formats:
            candidate = output_dir / f"{stem}.{ext}"
            if not candidate.exists():
                missing_variants.append(str(candidate))
    if missing_variants:
        print(
            "AVERTISSEMENT: sorties manquantes pour le test visuel:\n"
            + "\n".join(f"- {path}" for path in missing_variants)
        )
    else:
        formats_label = "/".join(fmt.upper() for fmt in formats)
        print(
            f"Test visuel: fichiers {formats_label} présents "
            f"pour {label} dans {output_dir}."
        )


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
        raise FileNotFoundError(f"CSV introuvable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    return (fieldnames, rows)


def _load_network_sizes_from_csvs(paths: list[Path]) -> list[int]:
    import pandas as pd

    sizes: set[int] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV introuvable: {path}")
        df = pd.read_csv(path)
        if "network_size" in df.columns:
            size_column = "network_size"
        elif "density" in df.columns:
            size_column = "density"
        else:
            raise ValueError(
                f"Le CSV {path} doit contenir une colonne "
                "'network_size' ou 'density'."
            )
        sizes.update(
            int(float(value))
            for value in df[size_column].dropna().unique().tolist()
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


def _suggest_step2_resume_command(expected_sizes: list[int]) -> str:
    sizes = " ".join(str(size) for size in expected_sizes)
    return (
        "python article_c/step2/run_step2.py --resume "
        f"--network-sizes {sizes} --replications 5 --seeds_base 1000"
    )


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
) -> tuple[bool, str]:
    if step not in cached_data:
        cached_data[step] = _load_csv_data(csv_path)
    fieldnames, rows = cached_data[step]
    if not fieldnames:
        print(
            "AVERTISSEMENT: "
            f"CSV vide pour {module_path}, figure ignorée."
        )
        return False, "CSV vide"
    sizes = _extract_network_sizes(csv_path)
    if len(sizes) < 2:
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        print(
            "Tailles détectées dans "
            f"{csv_path}: {sizes_label}."
        )
        print(
            "WARNING: "
            f"{module_path} nécessite au moins 2 tailles "
            "disponibles, figure ignorée."
        )
        print(f"CSV path: {csv_path}")
        return False, "moins de 2 tailles de réseau"
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
        return False, "colonnes algo/snir_mode manquantes"
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
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        print(
            "Tailles détectées dans "
            f"{csv_path}: {sizes_label}."
        )
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
        return False, "données incomplètes"
    return True, "OK"


@dataclass
class PlotStatus:
    step: str
    module_path: str
    status: str
    message: str


def _register_status(
    status_map: dict[str, PlotStatus],
    *,
    step: str,
    module_path: str,
    status: str,
    message: str,
) -> None:
    status_map[module_path] = PlotStatus(
        step=step,
        module_path=module_path,
        status=status,
        message=message,
    )


def _summarize_statuses(
    status_map: dict[str, PlotStatus],
    steps: list[str],
) -> dict[str, int]:
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    print("\nRésumé d'exécution des plots:")
    for step in steps:
        print(f"\n{step.upper()}:")
        for module_path in PLOT_MODULES[step]:
            entry = status_map.get(module_path)
            if entry is None:
                status_label = "SKIP"
                message = "Non exécuté."
            else:
                status_label = entry.status
                message = entry.message
            counts[status_label] = counts.get(status_label, 0) + 1
            print(f"- {module_path}: {status_label} ({message})")
    total = sum(counts.values())
    print(
        "\nBilan: "
        f"{counts['OK']} OK / {counts['FAIL']} FAIL / "
        f"{counts['SKIP']} SKIP (total {total})."
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    from article_c.common.plot_helpers import (
        parse_export_formats,
        set_default_export_formats,
    )

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    set_default_export_formats(export_formats)
    status_map: dict[str, PlotStatus] = {}
    invalid_modules = _validate_plot_modules_use_save_figure()
    if invalid_modules:
        for step, module_paths in PLOT_MODULES.items():
            for module_path in module_paths:
                if module_path in invalid_modules:
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=invalid_modules[module_path],
                    )
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    step1_results_dir = STEP1_RESULTS_DIR
    step2_results_dir = STEP2_RESULTS_DIR
    step1_csv = None
    step2_csv = None
    step_errors: dict[str, str] = {}
    if "step1" in steps:
        step1_csv = _ensure_step1_aggregated(step1_results_dir)
        if step1_csv is None:
            _report_missing_csv("Step1", step1_results_dir)
            step_errors["step1"] = "CSV Step1 manquant"
        else:
            _ensure_expected_results_dir(step1_csv, step1_results_dir, "Step1")
            if not (step1_results_dir / "done.flag").exists():
                print("AVERTISSEMENT: done.flag absent pour Step1, continuation.")
    if "step2" in steps:
        step2_csv = _ensure_step2_aggregated(step2_results_dir)
        if step2_csv is None:
            _report_missing_csv("Step2", step2_results_dir)
            step_errors["step2"] = "CSV Step2 manquant"
        else:
            _ensure_expected_results_dir(step2_csv, step2_results_dir, "Step2")
            if not (step2_results_dir / "done.flag").exists():
                print("AVERTISSEMENT: done.flag absent pour Step2, continuation.")
    if (
        step1_csv is not None
        and step2_csv is not None
        and step1_csv.resolve() == step2_csv.resolve()
    ):
        message = (
            "Step1 et Step2 pointent vers le même CSV agrégé. "
            "Vérifiez que chaque étape écrit dans son dossier results."
        )
        print(f"ERREUR: {message}")
        step_errors["step1"] = message
        step_errors["step2"] = message
    csv_paths: list[Path] = []
    if "step1" in steps and step1_csv is not None:
        csv_paths.append(step1_csv)
    if "step2" in steps and step2_csv is not None:
        csv_paths.append(step2_csv)
    step_network_sizes: dict[str, list[int]] = {}
    if "step1" in steps and step1_csv is not None:
        step_network_sizes["step1"] = _load_network_sizes_from_csvs([step1_csv])
    if "step2" in steps and step2_csv is not None:
        step_network_sizes["step2"] = _load_network_sizes_from_csvs([step2_csv])
    if args.network_sizes:
        network_sizes = args.network_sizes
        if not _validate_network_sizes(csv_paths, network_sizes):
            step_errors.setdefault(
                "step1",
                "validation des tailles de réseau échouée",
            )
            step_errors.setdefault(
                "step2",
                "validation des tailles de réseau échouée",
            )
            network_sizes = []
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
            step_errors.setdefault(
                "step1",
                "aucune taille de réseau détectée",
            )
            step_errors.setdefault(
                "step2",
                "aucune taille de réseau détectée",
            )
    skip_step2_plots = False
    if "step2" in steps:
        step2_sizes = step_network_sizes.get("step2", [])
        if len(step2_sizes) < 2:
            print(
                "WARNING: Step2 doit contenir au moins 2 tailles "
                "pour générer les plots RL. Aucun plot RL ne sera généré."
            )
            print(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
            expected_sizes = args.network_sizes or step_network_sizes.get(
                "step1", step2_sizes
            )
            if expected_sizes:
                print("Commande PowerShell pour terminer Step2 (mode reprise):")
                print(_suggest_step2_resume_command(expected_sizes))
            skip_step2_plots = True
        else:
            if args.network_sizes:
                expected_sizes = args.network_sizes
            elif step1_csv is not None and step1_csv.exists():
                expected_sizes = _load_network_sizes_from_csvs([step1_csv])
            else:
                expected_sizes = []
            if expected_sizes:
                missing_sizes = sorted(set(expected_sizes) - set(step2_sizes))
                if missing_sizes:
                    missing_label = ", ".join(str(size) for size in missing_sizes)
                    print(
                        "WARNING: Step2 ne contient pas toutes les tailles attendues."
                    )
                    print(f"Tailles attendues manquantes: {missing_label}")
                    print(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
                    print("Commande PowerShell pour terminer Step2 (mode reprise):")
                    print(_suggest_step2_resume_command(expected_sizes))
    csv_cache: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for step, module_paths in PLOT_MODULES.items():
        if step not in steps:
            continue
        if step in step_errors:
            for module_path in module_paths:
                if module_path not in status_map:
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="SKIP",
                        message=step_errors[step],
                    )
    if skip_step2_plots and "step2" in steps:
        for module_path in PLOT_MODULES["step2"]:
            if module_path not in status_map:
                _register_status(
                    status_map,
                    step="step2",
                    module_path=module_path,
                    status="SKIP",
                    message="Step2 incomplet (moins de 2 tailles)",
                )
    for step in steps:
        if step in step_errors:
            continue
        if step == "step2" and skip_step2_plots:
            continue
        for module_path in PLOT_MODULES[step]:
            if module_path in status_map and status_map[module_path].status == "FAIL":
                continue
            if step == "step1":
                if step1_csv is None:
                    continue
                csv_path = step1_csv
            else:
                if step2_csv is None:
                    continue
                csv_path = step2_csv
            rl10_network_sizes: list[int] | None = None
            if (
                step == "step2"
                and module_path.endswith("plot_RL10_reward_vs_pdr_scatter")
            ):
                if step1_csv is None or step2_csv is None:
                    continue
                step1_sizes = _load_network_sizes_from_csvs([step1_csv])
                step2_sizes = _load_network_sizes_from_csvs([step2_csv])
                intersection = sorted(set(step1_sizes) & set(step2_sizes))
                if len(intersection) < 2:
                    print(
                        "WARNING: "
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
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="SKIP",
                        message="moins de 2 tailles communes Step1/Step2",
                    )
                    continue
                rl10_network_sizes = intersection
            is_valid, reason = _validate_plot_data(
                step=step,
                module_path=module_path,
                csv_path=csv_path,
                cached_data=csv_cache,
            )
            if not is_valid:
                _register_status(
                    status_map,
                    step=step,
                    module_path=module_path,
                    status="SKIP",
                    message=reason,
                )
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
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    traceback.print_exc()
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=str(exc),
                    )
            else:
                try:
                    _run_plot_module(module_path, allow_sample=False)
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    traceback.print_exc()
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=str(exc),
                    )
    if "step1" in steps:
        _inspect_plot_outputs(
            ARTICLE_DIR / "step1" / "plots" / "output",
            "Step1",
            list(export_formats),
        )
    if "step2" in steps and not skip_step2_plots:
        _inspect_plot_outputs(
            ARTICLE_DIR / "step2" / "plots" / "output",
            "Step2",
            list(export_formats),
        )
    counts = _summarize_statuses(status_map, steps)
    if args.fail_on_error and counts.get("FAIL", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
